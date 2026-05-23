#include <iostream>
#include "mc_mini/geometry.hpp"
#include "mc_mini/material.hpp"
#include "mc_mini/particle.hpp"
#include "mc_mini/rng.hpp"
#include "mc_mini/tally.hpp"

namespace mcm {
    Particle sample_source_particle(Rng& rng) {
        Particle particle{};

        particle.x = 0.0;
        particle.y = 0.0;
        particle.z = 0.0;

        particle.energy = 1.0;
        particle.collisions = 0;
        particle.status = ParticleStatus::alive;

        rng.sample_isotropic_direction(particle.ux, particle.uy, particle.uz);

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
        Rng& rng
    ) {
        while (particle.is_alive()) {
            const double optical_depth = rng.sample_optical_depth();
            const double collision_distance = optical_depth / material.xs_t();
            const double boundary_distance = box.distance_to_boundary(particle);

            if (boundary_distance <= collision_distance) {
                move_particle(particle, boundary_distance);
                particle.status = ParticleStatus::escaped;
                return;
            }

            move_particle(particle, collision_distance);
            ++particle.collisions;

            if (rng.uniform01() < material.absorption_probability()) {
                particle.status = ParticleStatus::absorbed;
                return;
            }

            rng.sample_isotropic_direction(particle.ux, particle.uy, particle.uz);
        }
    }

    void score_particle(const Particle& particle, Tally& tally) {
        tally.collisions += particle.collisions;

        if (particle.status == ParticleStatus::absorbed) {
            ++tally.absorbed;
            return;
        }

        if (particle.status == ParticleStatus::escaped) {
            ++tally.escaped;
            return;
        }
    }
}

int main() {
    constexpr std::uint64_t particle_count = 10'000'000;

    mcm::Rng rng(12345);

    const mcm::Box box{
        .x_min = -1.0, .y_min = -1.0, .z_min = -1.0,
        .x_max = 1.0, .y_max = 1.0, .z_max = 1.0
    };

    const mcm::Material material{
        .xs_a = 0.2,
        .xs_s = 0.8
    };

    mcm::Tally tally{};

    for (std::uint64_t i = 0; i < particle_count; ++i) {
        mcm::Particle particle = mcm::sample_source_particle(rng);
        mcm::transport_history(particle, box, material, rng);
        mcm::score_particle(particle, tally);
    }

    std::cout << "Total: " << tally.total_particles() << '\n';
    std::cout << "Absorbed: " << tally.absorbed << '\n';
    std::cout << "Escaped: " << tally.escaped << '\n';
    std::cout << "Absorption fraction: " << tally.absorption_fraction() << '\n';
    std::cout << "Escape fraction: " << tally.escape_fraction() << '\n';
    std::cout << "Mean collisions: " << tally.mean_collisions() << '\n';

    return 0;
}
