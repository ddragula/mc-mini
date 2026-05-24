// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

#include "mc_mini/ace_loader.hpp"
#include "mc_mini/config.hpp"
#include "mc_mini/timer.hpp"

#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess) {                                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "        \
                      << cudaGetErrorString(err) << std::endl;                          \
            std::exit(EXIT_FAILURE);                                                    \
        }                                                                               \
    } while (0)

constexpr std::uint8_t PARTICLE_ALIVE = 0;
constexpr std::uint8_t PARTICLE_ABSORBED = 1;
constexpr std::uint8_t PARTICLE_ESCAPED = 2;
constexpr double PI = 3.141592653589793238462643383279502884;

struct DeviceBox {
    double x_min{};
    double y_min{};
    double z_min{};

    double x_max{};
    double y_max{};
    double z_max{};
};

struct DeviceParticles {
    double* x{};
    double* y{};
    double* z{};

    double* ux{};
    double* uy{};
    double* uz{};

    double* energy{};

    std::uint32_t* collisions{};
    std::uint8_t* status{};
};

struct DeviceEventData {
    double* collision_distance{};
    double* boundary_distance{};

    double* xs_a{};
    double* xs_s{};
    double* xs_t{};
};

struct DeviceXsTable {
    double* energies{};
    double* absorption{};
    double* elastic{};

    std::size_t size{};
    double atom_density{};
    double mass_number{};
};

struct DeviceCrossSections {
    double a{};
    double s{};
    double t{};
};

struct DeviceTally {
    double* escaped_energy{};
    double* absorbed_energy{};
    double* recoil_energy{};
    double* track_length{};
};

__device__ double clamp_double(double value, double min, double max) {
    return fmax(min, fmin(max, value));
}

__device__ double uniform01(curandState& state) {
    double value{};

    do {
        value = curand_uniform_double(&state);
    } while(value <= 0.0);

    return value;
}

__device__ void sample_isotropic_direction(
    curandState& state,
    double& ux,
    double& uy,
    double& uz
) {
    const double xi1 = uniform01(state);
    const double xi2 = uniform01(state);

    const double mu = 2.0 * xi1 - 1.0;
    const double phi = 2.0 * PI * xi2;

    const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));

    ux = sin_theta * cos(phi);
    uy = sin_theta * sin(phi);
    uz = mu;
}

__device__ void rotate_direction(
    DeviceParticles particles,
    std::uint32_t particle_index,
    double mu_lab,
    double phi
) {
    const double old_ux = particles.ux[particle_index];
    const double old_uy = particles.uy[particle_index];
    const double old_uz = particles.uz[particle_index];

    double helper_x = 0.0;
    double helper_y = 0.0;
    double helper_z = 1.0;

    if (fabs(old_uz) >= 0.999) {
        helper_x = 1.0;
        helper_z = 0.0;
    }

    double t1_x = helper_y * old_uz - helper_z * old_uy;
    double t1_y = helper_z * old_ux - helper_x * old_uz;
    double t1_z = helper_x * old_uy - helper_y * old_ux;

    const double t1_norm = sqrt(
        t1_x * t1_x +
        t1_y * t1_y +
        t1_z * t1_z
    );

    t1_x /= t1_norm;
    t1_y /= t1_norm;
    t1_z /= t1_norm;

    const double t2_x = old_uy * t1_z - old_uz * t1_y;
    const double t2_y = old_uz * t1_x - old_ux * t1_z;
    const double t2_z = old_ux * t1_y - old_uy * t1_x;

    mu_lab = clamp_double(mu_lab, -1.0, 1.0);

    const double sin_lab = sqrt(fmax(0.0, 1.0 - mu_lab * mu_lab));

    const double cos_phi = cos(phi);
    const double sin_phi = sin(phi);

    double new_ux =
        mu_lab * old_ux +
        sin_lab * (cos_phi * t1_x + sin_phi * t2_x);
    double new_uy =
        mu_lab * old_uy +
        sin_lab * (cos_phi * t1_y + sin_phi * t2_y);
    double new_uz =
        mu_lab * old_uz +
        sin_lab * (cos_phi * t1_z + sin_phi * t2_z);
    
    const double norm = sqrt(
        new_ux * new_ux +
        new_uy * new_uy +
        new_uz * new_uz
    );

    particles.ux[particle_index] = new_ux / norm;
    particles.uy[particle_index] = new_uy / norm;
    particles.uz[particle_index] = new_uz / norm;
}

__device__ void scatter_elastic_isotropic_cm(
    DeviceParticles particles,
    std::uint32_t particle_index,
    double mass_number,
    curandState& rng_state
) {
    const double a = mass_number;
    const double mu_cm = 2.0 * uniform01(rng_state) - 1.0;
    const double phi = 2.0 * PI * uniform01(rng_state);

    const double denominator = a * a + 2.0 * a * mu_cm + 1.0;
    const double energy_ratio = denominator / ((a + 1.0) * (a + 1.0));
    const double mu_lab = (1.0 + a * mu_cm) / sqrt(fmax(denominator, 1.0e-300));

    particles.energy[particle_index] *= energy_ratio;

    rotate_direction(particles, particle_index, mu_lab, phi);
}

__device__ double distance_to_boundary(
    DeviceParticles particles,
    std::uint32_t index,
    DeviceBox box
) {
    const double inf = INFINITY;

    const double tx = particles.ux[index] > 0.0
        ? (box.x_max - particles.x[index]) / particles.ux[index]
        : particles.ux[index] < 0.0
            ? (box.x_min - particles.x[index]) / particles.ux[index]
            : inf;
    
    const double ty = particles.uy[index] > 0.0
        ? (box.y_max - particles.y[index]) / particles.uy[index]
        : particles.uy[index] < 0.0
            ? (box.y_min - particles.y[index]) / particles.uy[index]
            : inf;

    const double tz = particles.uz[index] > 0.0
        ? (box.z_max - particles.z[index]) / particles.uz[index]
        : particles.uz[index] < 0.0
            ? (box.z_min - particles.z[index]) / particles.uz[index]
            : inf;
    
    return fmin(tx, fmin(ty, tz));
}

__device__ DeviceCrossSections lookup_cross_sections(
    DeviceXsTable table,
    double energy
) {
    if (energy <= table.energies[0]) {
        const double a = table.absorption[0] * table.atom_density;
        const double s = table.elastic[0] * table.atom_density;
        
        return DeviceCrossSections{a, s, a + s};
    }

    const std::size_t last = table.size - 1;

    if (energy >= table.energies[last]) {
        const double a = table.absorption[last] * table.atom_density;
        const double s = table.elastic[last] * table.atom_density;

        return DeviceCrossSections{a, s, a + s};
    }

    std::size_t left = 0;
    std::size_t right = last;

    while (right - left > 1) {
        const std::size_t mid = left + (right - left) / 2;

        if (table.energies[mid] > energy) {
            right = mid;
        } else {
            left = mid;
        }
    }

    const double e0 = table.energies[left];
    const double e1 = table.energies[right];

    const double t = (energy - e0) / (e1 - e0);

    const double micro_a =
        table.absorption[left] +
        t * (table.absorption[right] - table.absorption[left]);

    const double micro_s =
        table.elastic[left] +
        t * (table.elastic[right] - table.elastic[left]);
    
    const double a = micro_a * table.atom_density;
    const double s = micro_s * table.atom_density;

    return DeviceCrossSections{a, s, a + s};
}

__global__ void initialize_rng_kernel(
    curandState* rng_states,
    std::size_t particle_count,
    unsigned long long seed
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= particle_count) {
        return;
    }

    curand_init(
        seed,
        static_cast<unsigned long long>(i),
        0,
        &rng_states[i]
    );
}

__global__ void initialize_particles_kernel(
    DeviceParticles particles,
    std::uint32_t* active_queue,
    curandState* rng_states,
    std::size_t particle_count,
    double source_energy
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= particle_count) {
        return;
    }

    particles.x[i] = 0.0;
    particles.y[i] = 0.0;
    particles.z[i] = 0.0;

    curandState local_state = rng_states[i];
    sample_isotropic_direction(local_state, particles.ux[i], particles.uy[i], particles.uz[i]);
    rng_states[i] = local_state;

    particles.energy[i] = source_energy;
    particles.collisions[i] = 0;
    particles.status[i] = PARTICLE_ALIVE;

    active_queue[i] = static_cast<std::uint32_t>(i);
}

__global__ void classify_events_kernel(
    DeviceParticles particles,
    DeviceEventData event_data,
    DeviceXsTable xs_table,
    curandState* rng_states,
    std::uint32_t* active_queue,
    std::size_t active_count,
    std::uint32_t* escape_queue,
    std::uint32_t* escape_count,
    std::uint32_t* collision_queue,
    std::uint32_t* collision_count,
    DeviceBox box
) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= active_count) {
        return;
    }

    const std::uint32_t particle_index = active_queue[slot];

    const double boundary_distance = distance_to_boundary(particles, particle_index, box);

    const DeviceCrossSections xs = lookup_cross_sections(
        xs_table,
        particles.energy[particle_index]
    );

    event_data.xs_a[particle_index] = xs.a;
    event_data.xs_s[particle_index] = xs.s;
    event_data.xs_t[particle_index] = xs.t;

    curandState local_state = rng_states[particle_index];
    const double xi = uniform01(local_state);
    const double optical_depth = -log(xi);

    rng_states[particle_index] = local_state;

    const double collision_distance = optical_depth / xs.t;

    event_data.boundary_distance[particle_index] = boundary_distance;
    event_data.collision_distance[particle_index] = collision_distance;

    if (boundary_distance <= collision_distance) {
        const std::uint32_t position = atomicAdd(escape_count, 1u);
        escape_queue[position] = particle_index;
        return;
    }

    const std::uint32_t position = atomicAdd(collision_count, 1u);
    collision_queue[position] = particle_index;
}

__global__ void process_escapes_kernel(
    DeviceParticles particles,
    DeviceEventData event_data,
    DeviceTally tally,
    std::uint32_t* escape_queue,
    std::uint32_t escape_count
) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= escape_count) {
        return;
    }

    const std::uint32_t particle_index = escape_queue[slot];

    const double distance = event_data.boundary_distance[particle_index];
    atomicAdd(tally.track_length, distance);
    atomicAdd(tally.escaped_energy, particles.energy[particle_index]);

    particles.x[particle_index] += distance * particles.ux[particle_index];
    particles.y[particle_index] += distance * particles.uy[particle_index];
    particles.z[particle_index] += distance * particles.uz[particle_index];

    particles.status[particle_index] = PARTICLE_ESCAPED;
}

__global__ void process_collisions_kernel(
    DeviceParticles particles,
    DeviceEventData event_data,
    DeviceXsTable xs_table,
    DeviceTally tally,
    curandState* rng_states,
    std::uint32_t* collision_queue,
    std::uint32_t collision_count,
    std::uint32_t* next_active_queue,
    std::uint32_t* next_active_count
) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= collision_count) {
        return;
    }

    const std::uint32_t particle_index = collision_queue[slot];

    const double collision_distance = event_data.collision_distance[particle_index];
    atomicAdd(tally.track_length, collision_distance);

    particles.x[particle_index] += collision_distance * particles.ux[particle_index];
    particles.y[particle_index] += collision_distance * particles.uy[particle_index];
    particles.z[particle_index] += collision_distance * particles.uz[particle_index];

    particles.collisions[particle_index] += 1;

    const double energy_before_collision = particles.energy[particle_index];
    
    curandState local_state = rng_states[particle_index];

    const double xi = uniform01(local_state);

    const double absorption_probability =
        event_data.xs_a[particle_index] / event_data.xs_t[particle_index];

    if (xi < absorption_probability) {
        particles.status[particle_index] = PARTICLE_ABSORBED;
        atomicAdd(tally.absorbed_energy, energy_before_collision);
    } else {
        scatter_elastic_isotropic_cm(
            particles,
            particle_index,
            xs_table.mass_number,
            local_state
        );

        const double energy_after_collision = particles.energy[particle_index];
        atomicAdd(tally.recoil_energy, energy_before_collision - energy_after_collision);

        particles.status[particle_index] = PARTICLE_ALIVE;

        const std::uint32_t position = atomicAdd(next_active_count, 1u);
        next_active_queue[position] = particle_index;
    }

    rng_states[particle_index] = local_state;
}

int main(int argc, char* argv[]) {
    mcm::timer::record_start();

    const mcm::SimulationConfig config = mcm::load_simulation_config(argc, argv);
    const std::size_t particle_count = static_cast<std::size_t>(config.particle_count);
    constexpr double source_energy = 2.0; // MeV
    const mcm::Material material = mcm::load_ace_material_from_mass_density(
        config.material_file,
        config.mass_density
    );

    curandState* rng_states{};
    CUDA_CHECK(cudaMalloc(&rng_states, particle_count * sizeof(curandState)));

    DeviceXsTable xs_table{};
    xs_table.size = material.micro_xs.energies.size();
    xs_table.atom_density = material.atom_density;
    xs_table.mass_number = material.mass_number;

    CUDA_CHECK(cudaMalloc(&xs_table.energies, xs_table.size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&xs_table.absorption, xs_table.size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&xs_table.elastic, xs_table.size * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        xs_table.energies,
        material.micro_xs.energies.data(),
        xs_table.size * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        xs_table.absorption,
        material.micro_xs.absorption.data(),
        xs_table.size * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        xs_table.elastic,
        material.micro_xs.elastic.data(),
        xs_table.size * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    DeviceParticles particles{};
    DeviceEventData event_data{};
    DeviceTally tally{};

    CUDA_CHECK(cudaMalloc(&particles.x, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.y, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.z, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.ux, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.uy, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.uz, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.collisions, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&particles.energy, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.status, particle_count * sizeof(std::uint8_t)));

    CUDA_CHECK(cudaMalloc(&event_data.collision_distance, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&event_data.boundary_distance, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&event_data.xs_a, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&event_data.xs_s, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&event_data.xs_t, particle_count * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&tally.escaped_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tally.absorbed_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tally.recoil_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tally.track_length, sizeof(double)));

    CUDA_CHECK(cudaMemset(tally.escaped_energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(tally.absorbed_energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(tally.recoil_energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(tally.track_length, 0, sizeof(double)));

    std::uint32_t* active_queue{};
    CUDA_CHECK(cudaMalloc(&active_queue, particle_count * sizeof(std::uint32_t)));

    std::uint32_t* escape_queue{};
    std::uint32_t* collision_queue{};
    std::uint32_t* next_active_queue{};

    std::uint32_t* escape_count{};
    std::uint32_t* collision_count{};
    std::uint32_t* next_active_count{};

    CUDA_CHECK(cudaMalloc(&escape_queue, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&collision_queue, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&next_active_queue, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&escape_count, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&collision_count, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&next_active_count, sizeof(std::uint32_t)));

    // TODO: check if can be removed from here since they are also reset at the beginning of each iteration
    CUDA_CHECK(cudaMemset(escape_count, 0, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(collision_count, 0, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(next_active_count, 0, sizeof(std::uint32_t)));

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (particle_count + threads_per_block - 1) / threads_per_block
    );

    initialize_rng_kernel<<<blocks, threads_per_block>>>(
        rng_states,
        particle_count,
        12345ULL
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const double box_half_length = 0.5 * config.box_side_length;
    const DeviceBox box{
        .x_min = -box_half_length,
        .y_min = -box_half_length,
        .z_min = -box_half_length,
        .x_max = box_half_length,
        .y_max = box_half_length,
        .z_max = box_half_length
    };

    std::uint32_t active_count_host = static_cast<std::uint32_t>(particle_count);

    std::uint64_t total_escaped = 0;
    std::uint64_t total_absorbed = 0;
    std::uint64_t total_collisions = 0;

    std::uint32_t iteration = 0;
    constexpr std::uint32_t max_iterations = 10000;

    mcm::timer::record_initialization_end();

    initialize_particles_kernel<<<blocks, threads_per_block>>>(
        particles,
        active_queue,
        rng_states,
        particle_count,
        source_energy
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    while (active_count_host > 0 && iteration < max_iterations) {
        CUDA_CHECK(cudaMemset(escape_count, 0, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemset(collision_count, 0, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemset(next_active_count, 0, sizeof(std::uint32_t)));

        const int active_blocks = static_cast<int>(
            (active_count_host + threads_per_block - 1) / threads_per_block
        );

        classify_events_kernel<<<active_blocks, threads_per_block>>>(
            particles,
            event_data,
            xs_table,
            rng_states,
            active_queue,
            active_count_host,
            escape_queue,
            escape_count,
            collision_queue,
            collision_count,
            box
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::uint32_t escape_count_host{};
        std::uint32_t collision_count_host{};

        CUDA_CHECK(cudaMemcpy(
            &escape_count_host,
            escape_count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            &collision_count_host,
            collision_count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ));

        const int escape_blocks = static_cast<int>(
            (escape_count_host + threads_per_block - 1) / threads_per_block
        );
        const int collision_blocks = static_cast<int>(
            (collision_count_host + threads_per_block - 1) / threads_per_block
        );

        if (escape_count_host > 0) {
            process_escapes_kernel<<<escape_blocks, threads_per_block>>>(
                particles,
                event_data,
                tally,
                escape_queue,
                escape_count_host
            );
        }

        if (collision_count_host > 0) {
            process_collisions_kernel<<<collision_blocks, threads_per_block>>>(
                particles,
                event_data,
                xs_table,
                tally,
                rng_states,
                collision_queue,
                collision_count_host,
                next_active_queue,
                next_active_count
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::uint32_t next_active_count_host{};

        CUDA_CHECK(cudaMemcpy(
            &next_active_count_host,
            next_active_count,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ));

        total_escaped += escape_count_host;
        total_collisions += collision_count_host;
        total_absorbed +=
            static_cast<std::uint64_t>(collision_count_host) -
            static_cast<std::uint64_t>(next_active_count_host);

        std::swap(active_queue, next_active_queue);
        active_count_host = next_active_count_host;

        ++iteration;
    }

    mcm::timer::record_transport_end();

    std::cout << "iterations: " << iteration << '\n';
    std::cout << "active left: " << active_count_host << '\n';
    std::cout << "total escaped: " << total_escaped << '\n';
    std::cout << "total absorbed: " << total_absorbed << '\n';
    std::cout << "total collisions: " << total_collisions << '\n';
    std::cout << "terminated total: "
            << total_escaped + total_absorbed
            << '\n';
    
    // === TALLY RESULTS ===

    double escaped_energy_host{};
    double absorbed_energy_host{};
    double recoil_energy_host{};
    double track_length_host{};

    CUDA_CHECK(cudaMemcpy(
        &escaped_energy_host,
        tally.escaped_energy,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        &absorbed_energy_host,
        tally.absorbed_energy,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        &recoil_energy_host,
        tally.recoil_energy,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        &track_length_host,
        tally.track_length,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    const double source_energy_total = static_cast<double>(particle_count) * source_energy;
    const double accounted_energy = escaped_energy_host + absorbed_energy_host + recoil_energy_host;
    const double volume = (box.x_max - box.x_min) * (box.y_max - box.y_min) * (box.z_max - box.z_min);

    std::cout << '\n';

    std::cout << "Source energy: " << source_energy_total << '\n';
    std::cout << "Escaped energy: " << escaped_energy_host << '\n';
    std::cout << "Absorbed energy: " << absorbed_energy_host << '\n';
    std::cout << "Recoil energy: " << recoil_energy_host << '\n';
    std::cout << "Accounted energy: " << accounted_energy << '\n';
    std::cout << "Energy balance error: "
        << source_energy_total - accounted_energy
        << '\n';

    std::cout << '\n';

    std::cout << "Mean energy of escaped particles: "
        << escaped_energy_host / static_cast<double>(total_escaped)
        << '\n';

    std::cout << "Track length: " << track_length_host << '\n';
    std::cout << "Cell volume: " << volume << '\n';
    std::cout << "Track-length flux: "
        << track_length_host / (static_cast<double>(particle_count) * volume)
        << std::endl << std::endl;

    mcm::timer::print_timing_results(particle_count);

    // === MEMORY CLEANUP ===

    CUDA_CHECK(cudaFree(particles.x));
    CUDA_CHECK(cudaFree(particles.y));
    CUDA_CHECK(cudaFree(particles.z));
    CUDA_CHECK(cudaFree(particles.ux));
    CUDA_CHECK(cudaFree(particles.uy));
    CUDA_CHECK(cudaFree(particles.uz));
    CUDA_CHECK(cudaFree(particles.energy));
    CUDA_CHECK(cudaFree(particles.status));
    CUDA_CHECK(cudaFree(particles.collisions));
    CUDA_CHECK(cudaFree(active_queue));
    CUDA_CHECK(cudaFree(escape_queue));
    CUDA_CHECK(cudaFree(collision_queue));
    CUDA_CHECK(cudaFree(next_active_queue));
    CUDA_CHECK(cudaFree(escape_count));
    CUDA_CHECK(cudaFree(collision_count));
    CUDA_CHECK(cudaFree(next_active_count));
    CUDA_CHECK(cudaFree(event_data.collision_distance));
    CUDA_CHECK(cudaFree(event_data.boundary_distance));
    CUDA_CHECK(cudaFree(event_data.xs_a));
    CUDA_CHECK(cudaFree(event_data.xs_s));
    CUDA_CHECK(cudaFree(event_data.xs_t));
    CUDA_CHECK(cudaFree(xs_table.energies));
    CUDA_CHECK(cudaFree(xs_table.absorption));
    CUDA_CHECK(cudaFree(xs_table.elastic));
    CUDA_CHECK(cudaFree(rng_states));
    CUDA_CHECK(cudaFree(tally.escaped_energy));
    CUDA_CHECK(cudaFree(tally.absorbed_energy));
    CUDA_CHECK(cudaFree(tally.recoil_energy));
    CUDA_CHECK(cudaFree(tally.track_length));

    return 0;
}
