// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

#include "mc_mini/xs_table.hpp"

namespace mcm {

    struct CrossSections {
        double a{}; // absorption
        double s{}; // scattering
        double t{}; // total (calculated from absorption + scattering)

        double ace_total{}; // ACE total, for diagnostics only
    };

    struct Material {
        double atom_density{};
        double mass_number{1.0};
        XsTable micro_xs;

        void validate() const {
            if (atom_density <= 0.0) {
                throw std::runtime_error("Material: atom_density must be positive");
            }

            if (mass_number <= 0.0) {
                throw std::runtime_error("Material: mass_number must be positive");
            }

            micro_xs.validate("Material.micro_xs");
        }

        [[nodiscard]] CrossSections cross_sections(double energy) const {
            const XsValues micro = micro_xs.value_at(energy);

            const double a = micro.absorption * atom_density;
            const double s = micro.elastic * atom_density;

            return CrossSections{
                .a = a,
                .s = s,
                .t = a + s,
                .ace_total = micro.total * atom_density
            };
        }
    };

}
