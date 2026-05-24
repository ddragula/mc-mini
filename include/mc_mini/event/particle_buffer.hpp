// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mc_mini/particle.hpp"

namespace mcm::event {

    struct ParticleBuffer {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;

        std::vector<double> ux;
        std::vector<double> uy;
        std::vector<double> uz;

        std::vector<double> energy;

        std::vector<std::uint32_t> collisions;
        std::vector<ParticleStatus> status;

        void resize(std::size_t count) {
            x.resize(count);
            y.resize(count);
            z.resize(count);

            ux.resize(count);
            uy.resize(count);
            uz.resize(count);

            energy.resize(count, 1.0);

            collisions.resize(count);
            status.resize(count, ParticleStatus::alive);
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return energy.size();
        }

        [[nodiscard]] bool is_alive(std::size_t index) const noexcept {
            return status[index] == ParticleStatus::alive;
        }
    };

}
