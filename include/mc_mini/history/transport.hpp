// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mc_mini/geometry.hpp"
#include "mc_mini/material.hpp"
#include "mc_mini/particle.hpp"
#include "mc_mini/rng.hpp"
#include "mc_mini/scattering.hpp"
#include "mc_mini/tally.hpp"

namespace mcm {

    inline Particle sample_source_particle(Rng& rng) {
        Particle particle;

        particle.energy = 2.0; // MeV

        rng.sample_isotropic_direction(
            particle.ux,
            particle.uy,
            particle.uz
        );

        return particle;
    }

    inline void move_particle(Particle& particle, double distance) noexcept {
        particle.x += distance * particle.ux;
        particle.y += distance * particle.uy;
        particle.z += distance * particle.uz;
    }

    inline void transport_history(
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
