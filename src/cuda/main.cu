#include <cuda_runtime.h>

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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

__global__ void initialize_particles_kernel(
    DeviceParticles particles,
    std::uint32_t* active_queue,
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

    // Deterministic direction for now.
    // Real isotropic RNG comes later. (TODO)
    particles.ux[i] = 1.0;
    particles.uy[i] = 0.0;
    particles.uz[i] = 0.0;

    particles.energy[i] = source_energy;
    particles.collisions[i] = 0;
    particles.status[i] = PARTICLE_ALIVE;

    active_queue[i] = static_cast<std::uint32_t>(i);
}

__global__ void classify_events_kernel(
    DeviceParticles particles,
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
    const double collision_distance = particle_index % 2 == 0 ? 3.0 : 7.0; // Placeholder for now. (TODO)

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
    std::uint32_t* escape_queue,
    std::uint32_t escape_count,
    DeviceBox box
) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= escape_count) {
        return;
    }

    const std::uint32_t particle_index = escape_queue[slot];

    const double distance = distance_to_boundary(particles, particle_index, box);

    particles.x[particle_index] += distance * particles.ux[particle_index];
    particles.y[particle_index] += distance * particles.uy[particle_index];
    particles.z[particle_index] += distance * particles.uz[particle_index];

    particles.status[particle_index] = PARTICLE_ESCAPED;
}

__global__ void process_collisions_kernel(
    DeviceParticles particles,
    std::uint32_t* collision_queue,
    std::uint32_t collision_count
) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= collision_count) {
        return;
    }

    const std::uint32_t particle_index = collision_queue[slot];

    constexpr double collision_distance = 3.0; // Placeholder for now. (TODO)

    particles.x[particle_index] += collision_distance * particles.ux[particle_index];
    particles.y[particle_index] += collision_distance * particles.uy[particle_index];
    particles.z[particle_index] += collision_distance * particles.uz[particle_index];

    particles.collisions[particle_index] += 1;
    particles.status[particle_index] = PARTICLE_ABSORBED;
}

int main() {
    constexpr std::size_t particle_count = 1'000'000;
    constexpr double source_energy = 2.0; // MeV

    DeviceParticles particles{};

    CUDA_CHECK(cudaMalloc(&particles.x, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.y, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.z, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.ux, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.uy, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.uz, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.collisions, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&particles.energy, particle_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&particles.status, particle_count * sizeof(std::uint8_t)));

    std::uint32_t* active_queue{};
    CUDA_CHECK(cudaMalloc(&active_queue, particle_count * sizeof(std::uint32_t)));

    std::uint32_t* escape_queue{};
    std::uint32_t* collision_queue{};

    std::uint32_t* escape_count{};
    std::uint32_t* collision_count{};

    CUDA_CHECK(cudaMalloc(&escape_queue, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&collision_queue, particle_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&escape_count, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&collision_count, sizeof(std::uint32_t)));

    CUDA_CHECK(cudaMemset(escape_count, 0, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(collision_count, 0, sizeof(std::uint32_t)));

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (particle_count + threads_per_block - 1) / threads_per_block
    );

    initialize_particles_kernel<<<blocks, threads_per_block>>>(
        particles,
        active_queue,
        particle_count,
        source_energy
    );

    const DeviceBox box{
        .x_min = -5.0,
        .y_min = -5.0,
        .z_min = -5.0,
        .x_max = 5.0,
        .y_max = 5.0,
        .z_max = 5.0
    };

    classify_events_kernel<<<blocks, threads_per_block>>>(
        particles,
        active_queue,
        particle_count,
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

    process_escapes_kernel<<<escape_blocks, threads_per_block>>>(
        particles,
        escape_queue,
        escape_count_host,
        box
    );

    process_collisions_kernel<<<collision_blocks, threads_per_block>>>(
        particles,
        collision_queue,
        collision_count_host
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "escape count: " << escape_count_host << '\n';
    std::cout << "collision count: " << collision_count_host << '\n';
    std::cout << "classified total: " << escape_count_host + collision_count_host << '\n';

    std::vector<double> x_host(10);
    std::vector<std::uint8_t> status_host(10);
    std::vector<std::uint32_t> collisions_host(10);

    CUDA_CHECK(cudaMemcpy(
        x_host.data(),
        particles.x,
        x_host.size() * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        status_host.data(),
        particles.status,
        status_host.size() * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        collisions_host.data(),
        particles.collisions,
        collisions_host.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost
    ));

    std::cout << "first x: ";
    for (double value : x_host) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    std::cout << "first statuses: ";
    for (std::uint8_t value : status_host) {
        std::cout << static_cast<int>(value) << ' ';
    }
    std::cout << '\n';
    
    std::cout << "first collisions: ";
    for (std::uint32_t value : collisions_host) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

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
    CUDA_CHECK(cudaFree(escape_count));
    CUDA_CHECK(cudaFree(collision_count));

    return 0;
}
