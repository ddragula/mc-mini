// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "mc_mini/ace_loader.hpp"
#include "mc_mini/config.hpp"
#include "mc_mini/geometry.hpp"
#include "mc_mini/history/transport.hpp"
#include "mc_mini/material.hpp"
#include "mc_mini/rng.hpp"
#include "mc_mini/tally.hpp"
#include "mc_mini/timer.hpp"

namespace {

    struct alignas(64) WorkerTally {
        mcm::Tally tally{};
    };

    [[nodiscard]] std::uint64_t requested_thread_count(
        std::uint64_t particle_count
    ) {
        if (const char* value = std::getenv("MC_MINI_THREADS")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);

            if (end != value && parsed > 0) {
                return std::min<std::uint64_t>(parsed, particle_count);
            }
        }

        const unsigned int hardware_threads = std::thread::hardware_concurrency();

        if (hardware_threads == 0) {
            return 1;
        }

        return std::min<std::uint64_t>(hardware_threads, particle_count);
    }

    [[nodiscard]] std::uint64_t seed_for_worker(std::uint64_t worker_index) {
        constexpr std::uint64_t base_seed = 12345;
        constexpr std::uint64_t splitmix_increment = 0x9e3779b97f4a7c15ULL;

        return base_seed + splitmix_increment * (worker_index + 1);
    }

    void merge_tally(mcm::Tally& total, const mcm::Tally& local) noexcept {
        total.absorbed += local.absorbed;
        total.escaped += local.escaped;
        total.collisions += local.collisions;

        total.source_energy += local.source_energy;
        total.escaped_energy += local.escaped_energy;
        total.absorbed_energy += local.absorbed_energy;
        total.recoil_energy += local.recoil_energy;
        total.track_length += local.track_length;
    }

}

int main(int argc, char* argv[]) {
    mcm::timer::record_start();

    const mcm::SimulationConfig config = mcm::load_simulation_config(argc, argv);
    const std::uint64_t particle_count = config.particle_count;

    const mcm::Box box = config.make_centered_box();

    const mcm::Material material = mcm::load_ace_material_from_mass_density(
        config.material_file,
        config.mass_density
    );
    const std::uint64_t thread_count = requested_thread_count(particle_count);

    std::vector<std::thread> workers;
    std::vector<WorkerTally> worker_tallies(thread_count);

    workers.reserve(thread_count);

    mcm::timer::record_initialization_end();

    const std::uint64_t histories_per_thread = particle_count / thread_count;
    const std::uint64_t remainder = particle_count % thread_count;

    for (std::uint64_t worker_index = 0; worker_index < thread_count; ++worker_index) {
        const std::uint64_t local_particle_count =
            histories_per_thread + (worker_index < remainder ? 1 : 0);

        workers.emplace_back(
            [&, worker_index, local_particle_count] {
                mcm::Rng rng(seed_for_worker(worker_index));
                mcm::Tally& tally = worker_tallies[worker_index].tally;

                for (std::uint64_t i = 0; i < local_particle_count; ++i) {
                    mcm::Particle particle = mcm::sample_source_particle(rng);
                    mcm::transport_history(particle, box, material, rng, tally);
                }
            }
        );
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    mcm::Tally tally{};

    for (const WorkerTally& worker_tally : worker_tallies) {
        merge_tally(tally, worker_tally.tally);
    }

    mcm::timer::record_transport_end();

    std::cout << "==== SIMULATION RESULTS =====" << std::endl << std::endl;

    std::cout << "Threads: " << thread_count << '\n';
    std::cout << "Total: " << tally.total_particles() << '\n';
    std::cout << "Absorbed: " << tally.absorbed << '\n';
    std::cout << "Escaped: " << tally.escaped << '\n';
    std::cout << "Absorption fraction: " << tally.absorption_fraction() << '\n';
    std::cout << "Escape fraction: " << tally.escape_fraction() << '\n';
    std::cout << "Mean collisions: " << tally.mean_collisions() << std::endl;
    std::cout << std::endl;

    std::cout << "Source energy: " << tally.source_energy << '\n';
    std::cout << "Escaped energy: " << tally.escaped_energy << '\n';
    std::cout << "Absorbed energy: " << tally.absorbed_energy << '\n';
    std::cout << "Recoil energy: " << tally.recoil_energy << '\n';
    std::cout << "Accounted energy: " << tally.accounted_energy() << '\n';
    std::cout << "Energy balance error: " << tally.source_energy - tally.accounted_energy() << std::endl;
    std::cout << std::endl;

    std::cout << "Mean energy of escaped particles: " <<
        tally.escaped_energy / static_cast<double>(tally.escaped) <<
        std::endl << std::endl;

    std::cout << "Track length: " << tally.track_length << '\n';
    std::cout << "Cell volume: " << box.volume() << '\n';
    std::cout << "Track-length flux: " << tally.track_length_flux(particle_count, box.volume()) << std::endl;
    std::cout << std::endl;

    mcm::timer::print_timing_results(particle_count);

    return 0;
}
