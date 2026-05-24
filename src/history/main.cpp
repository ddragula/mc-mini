// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mc_mini/config.hpp"
#include "mc_mini/history/transport.hpp"
#include "mc_mini/geometry.hpp"
#include "mc_mini/material.hpp"
#include "mc_mini/rng.hpp"
#include "mc_mini/tally.hpp"
#include "mc_mini/ace_loader.hpp"
#include "mc_mini/timer.hpp"

int main(int argc, char* argv[]) {
    mcm::timer::record_start();

    const mcm::SimulationConfig config = mcm::load_simulation_config(argc, argv);
    const std::uint64_t particle_count = config.particle_count;

    mcm::Rng rng(12345);

    const mcm::Box box = config.make_centered_box();

    const mcm::Material material = mcm::load_ace_material_from_mass_density(
        config.material_file,
        config.mass_density
    );

    mcm::Tally tally{};

    mcm::timer::record_initialization_end();

    for (std::uint64_t i = 0; i < particle_count; ++i) {
        mcm::Particle particle = mcm::sample_source_particle(rng);
        mcm::transport_history(particle, box, material, rng, tally);
    }

    mcm::timer::record_transport_end();

    std::cout << "==== SIMULATION RESULTS =====" << std::endl << std::endl;

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
