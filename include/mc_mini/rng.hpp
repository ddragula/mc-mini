// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace mcm {

    inline constexpr double pi = 3.141592653589793238462643383279502884;

    class Rng {
    public:
        explicit Rng(std::uint64_t seed) :
            engine_(seed),
            uniform_distribution_(0.0, 1.0) {}

        [[nodiscard]] double uniform01() {
            double value{};

            do {
                value = uniform_distribution_(engine_);
            } while (value <= 0.0);

            return value;
        }

        [[nodiscard]] double sample_optical_depth() {
            return -std::log(uniform01());
        }

        void sample_isotropic_direction(
            double& ux,
            double& uy,
            double& uz
        ) {
            const double mu = 2.0 * uniform01() - 1.0;
            const double phi = 2.0 * pi * uniform01();

            const double sin_theta = std::sqrt(1.0 - mu * mu);

            ux = sin_theta * std::cos(phi);
            uy = sin_theta * std::sin(phi);
            uz = mu;
        }

    private:
        std::mt19937_64 engine_;
        std::uniform_real_distribution<double> uniform_distribution_;
    };

}
