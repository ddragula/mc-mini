// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mc_mini/geometry.hpp"
#include "mc_mini/material.hpp"
#include "mc_mini/particle.hpp"
#include "mc_mini/rng.hpp"
#include "mc_mini/tally.hpp"
#include "mc_mini/scattering.hpp"
#include "mc_mini/ace_loader.hpp"
#include "mc_mini/timer.hpp"

namespace mcm {
    Particle sample_source_particle(Rng& rng) {
        Particle particle;

        particle.energy = 2.0; // MeV

        rng.sample_isotropic_direction(
            particle.ux,
            particle.uy,
            particle.uz
        );

        return particle;
    }

    void move_particle(Particle& particle, double distance) noexcept {
        particle.x += distance * particle.ux;
        particle.y += distance * particle.uy;
        particle.z += distance * particle.uz;
    }

    void transport_history(
        Particle& particle,
        const Box& box,
        const Material& material,
        Rng& rng,
        Tally& tally
    ) {
        tally.source_energy += particle.energy;

        while (particle.is_alive()) {
            const CrossSections xs = material.cross_sections(particle.energy);

            const double optical_depth = rng.sample_optical_depth();
            const double collision_distance = optical_depth / xs.t;
            const double boundary_distance = box.distance_to_boundary(particle);

            if (boundary_distance <= collision_distance) {
                tally.score_track_length(boundary_distance);
                move_particle(particle, boundary_distance);

                particle.status = ParticleStatus::escaped;
                tally.escaped += 1;
                tally.escaped_energy += particle.energy;

                return;
            }

            tally.score_track_length(collision_distance);
            move_particle(particle, collision_distance);

            ++particle.collisions;
            ++tally.collisions;

            if (rng.uniform01() < xs.a / xs.t) {
                particle.status = ParticleStatus::absorbed;
                tally.absorbed += 1;
                tally.absorbed_energy += particle.energy;

                return;
            }

            const double energy_before_scattering = particle.energy;

            scatter_elastic_isotropic_cm(particle, material, rng);

            tally.recoil_energy += energy_before_scattering - particle.energy;
        }
    }
}

int main() {
    mcm::timer::record_start();

    constexpr std::uint64_t particle_count = 10'000'000;

    mcm::Rng rng(12345);

    const mcm::Box box{
        .x_min = -5.0, .y_min = -5.0, .z_min = -5.0,
        .x_max = 5.0, .y_max = 5.0, .z_max = 5.0
    };

    const mcm::Material material = mcm::load_ace_material_from_mass_density("data/ace/C0.ACE", 2.26);

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
