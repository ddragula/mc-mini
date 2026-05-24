// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace mcm {

    enum class ParticleStatus : std::uint8_t {
        alive,
        absorbed,
        escaped
    };

    struct Particle {
        double x{}, y{}, z{};
        double ux{}, uy{}, uz{};
        double energy{1.0};

        std::uint32_t collisions{};
        ParticleStatus status{ParticleStatus::alive};

        [[nodiscard]] bool is_alive() const noexcept {
            return status == ParticleStatus::alive;
        }
    };

}
