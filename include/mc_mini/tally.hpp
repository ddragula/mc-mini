#pragma once

#include <cstdint>

namespace mcm {

    struct Tally {
        std::uint64_t absorbed{};
        std::uint64_t escaped{};
        std::uint64_t collisions{};

        double source_energy{};
        double escaped_energy{};
        double absorbed_energy{};
        double recoil_energy{};
        double track_length{};

        void score_track_length(double distance) noexcept {
            track_length += distance;
        }

        [[nodiscard]] std::uint64_t total_particles() const noexcept {
            return absorbed + escaped;
        }

        [[nodiscard]] double absorption_fraction() const noexcept {
            return static_cast<double>(absorbed) / static_cast<double>(total_particles());
        }

        [[nodiscard]] double escape_fraction() const noexcept {
            return static_cast<double>(escaped) / static_cast<double>(total_particles());
        }

        [[nodiscard]] double mean_collisions() const noexcept {
            return static_cast<double>(collisions) / static_cast<double>(total_particles());
        }

        [[nodiscard]] double accounted_energy() const noexcept {
            return escaped_energy + absorbed_energy + recoil_energy;
        }

        [[nodiscard]] double track_length_flux(
            std::uint64_t source_particles,
            double volume
        ) const noexcept {
            return track_length / (static_cast<double>(source_particles) * volume);
        }
    };

}
