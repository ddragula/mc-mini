// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "mc_mini/material.hpp"
#include "mc_mini/particle.hpp"
#include "mc_mini/rng.hpp"

namespace mcm {

    inline void rotate_direction(
        Particle& particle,
        double mu_lab,
        double phi
    ) noexcept {
        const double old_ux = particle.ux;
        const double old_uy = particle.uy;
        const double old_uz = particle.uz;

        const bool use_z_axis = std::abs(old_uz) > 0.999;
        const double helper_x = 0.0;
        const double helper_y = use_z_axis ? 1.0 : 0.0;
        const double helper_z = use_z_axis ? 0.0 : 1.0;

        double t1_x = helper_y * old_uz - helper_z * old_uy;
        double t1_y = helper_z * old_ux - helper_x * old_uz;
        double t1_z = helper_x * old_uy - helper_y * old_ux;

        const double t1_norm = std::sqrt(
            t1_x * t1_x +
            t1_y * t1_y +
            t1_z * t1_z
        );

        t1_x /= t1_norm;
        t1_y /= t1_norm;
        t1_z /= t1_norm;

        const double t2_x = old_uy * t1_z - old_uz * t1_y;
        const double t2_y = old_uz * t1_x - old_ux * t1_z;
        const double t2_z = old_ux * t1_y - old_uy * t1_x;

        const double sin_lab = std::sqrt(
            std::max(0.0, 1.0 - mu_lab * mu_lab)
        );

        const double cos_phi = std::cos(phi);
        const double sin_phi = std::sin(phi);

        particle.ux =
            mu_lab * old_ux +
            sin_lab * (cos_phi * t1_x + sin_phi * t2_x);

        particle.uy =
            mu_lab * old_uy +
            sin_lab * (cos_phi * t1_y + sin_phi * t2_y);
        
        particle.uz =
            mu_lab * old_uz +
            sin_lab * (cos_phi * t1_z + sin_phi * t2_z);
    }

    inline void scatter_elastic_isotropic_cm(
        Particle& particle,
        const Material& material,
        Rng& rng
    ) noexcept {
        const double a = material.mass_number;
        const double mu_cm = 2.0 * rng.uniform01() - 1.0;
        const double phi = 2.0 * M_PI * rng.uniform01();

        const double energy_ratio =
            (a * a + 2.0 * a * mu_cm + 1.0) /
            ((a + 1.0) * (a + 1.0));
        
        const double mu_lab =
            (1.0 + a * mu_cm) /
            std::sqrt(a * a + 2.0 * a * mu_cm + 1.0);

        particle.energy *= energy_ratio;

        rotate_direction(
            particle,
            std::clamp(mu_lab, -1.0, 1.0),
            phi
        );
    }

}
