#pragma once

#include <cstdint>

namespace mcm {

    struct Tally {
        std::uint64_t absorbed{};
        std::uint64_t escaped{};
        std::uint64_t collisions{};

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
    };

}
