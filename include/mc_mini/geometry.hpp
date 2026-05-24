// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <limits>

#include "mc_mini/particle.hpp"

namespace mcm {

    struct Box {
        double x_min{}, y_min{}, z_min{};
        double x_max{}, y_max{}, z_max{};

        [[nodiscard]] double distance_to_boundary(const Particle& particle) const noexcept {
            const double inf = std::numeric_limits<double>::infinity();

            const double tx = particle.ux > 0.0
                ? (x_max - particle.x) / particle.ux
                : particle.ux < 0.0
                    ? (x_min - particle.x) / particle.ux
                    : inf;
            const double ty = particle.uy > 0.0
                ? (y_max - particle.y) / particle.uy
                : particle.uy < 0.0
                    ? (y_min - particle.y) / particle.uy
                    : inf;
            const double tz = particle.uz > 0.0
                ? (z_max - particle.z) / particle.uz
                : particle.uz < 0.0
                    ? (z_min - particle.z) / particle.uz
                    : inf;

            return std::min({tx, ty, tz});
        }

        [[nodiscard]] double volume() const noexcept {
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min);
        }
    };

}