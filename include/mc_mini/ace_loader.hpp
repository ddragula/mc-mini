// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mc_mini/material.hpp"

namespace mcm {

    struct AceHeader {
        std::string zaid;
        double atomic_weight_ratio{};
        double temperature{};
        std::string date;
    };

    struct AceFile {
        AceHeader header{};
        std::array<int, 16> nxs{};
        std::array<int, 32> jxs{};
        std::vector<double> xss;
    };

    inline AceFile read_ascii_ace(const std::string& path) {
        std::ifstream file{path};

        if (!file) {
            throw std::runtime_error("Failed to open ACE file: " + path);
        }

        AceFile ace{};

        if (!(
            file >> ace.header.zaid
                 >> ace.header.atomic_weight_ratio
                 >> ace.header.temperature
                 >> ace.header.date
        )) {
            throw std::runtime_error("Failed to read ACE header from: " + path);
        }

        std::string rest_of_line{};
        std::getline(file, rest_of_line);

        std::string title_line{};
        std::getline(file, title_line);

        // ACE has 16 IZ/AW pairs here, i.e. 32 numeric tokens.
        // We do not need them for the current transport model.
        std::string ignored_token{};

        for (int i = 0; i < 32; ++i) {
            if (!(file >> ignored_token)) {
                throw std::runtime_error("Failed to skip ACE IZ/AW block");
            }
        }

        for (int i = 0; i < 16; ++i) {
            if (!(file >> ace.nxs[static_cast<std::size_t>(i)])) {
                throw std::runtime_error("Failed to read ACE NXS block");
            }
        }

        for (int i = 0; i < 32; ++i) {
            if (!(file >> ace.jxs[static_cast<std::size_t>(i)])) {
                throw std::runtime_error("Failed to read ACE JXS block");
            }
        }

        const int xss_length = ace.nxs[0];

        if (xss_length <= 0) {
            throw std::runtime_error("ACE NXS[0] must be positive");
        }

        ace.xss.resize(static_cast<std::size_t>(xss_length));

        for (std::size_t i = 0; i < ace.xss.size(); ++i) {
            if (!(file >> ace.xss[i])) {
                throw std::runtime_error("Failed to read ACE XSS block");
            }
        }

        return ace;
    }

    inline XsTable extract_esz_table(const AceFile& ace) {
        const int xss_length = ace.nxs[0];
        const int energy_count = ace.nxs[2];

        if (xss_length <= 0) {
            throw std::runtime_error("ACE NXS[0] must be positive");
        }

        if (energy_count <= 0) {
            throw std::runtime_error("ACE NXS[2] / NES must be positive");
        }

        if (ace.jxs[0] <= 0) {
            throw std::runtime_error("ACE JXS[0] / ESZ pointer must be positive");
        }

        const std::size_t n = static_cast<std::size_t>(energy_count);

        // ACE uses 1-based indexing; C++ uses 0-based indexing.
        const std::size_t esz_start = static_cast<std::size_t>(ace.jxs[0] - 1);

        const std::size_t required_size = esz_start + 5 * n;

        if (required_size > ace.xss.size()) {
            throw std::runtime_error("ACE ESZ block exceeds XSS size");
        }

        XsTable table{};
        table.energies.resize(n);
        table.total.resize(n);
        table.absorption.resize(n);
        table.elastic.resize(n);
        table.heating.resize(n);

        for (std::size_t i = 0; i < n; ++i) {
            table.energies[i] = ace.xss[esz_start + i];
            table.total[i] = ace.xss[esz_start + n + i];
            table.absorption[i] = ace.xss[esz_start + 2 * n + i];
            table.elastic[i] = ace.xss[esz_start + 3 * n + i];
            table.heating[i] = ace.xss[esz_start + 4 * n + i];
        }

        table.validate("ACE ESZ");

        return table;
    }

    inline double atom_density_from_mass_density(
        double mass_density_g_per_cm3,
        double atomic_mass_g_per_mol
    ) {
        constexpr double avogadro = 6.02214076e23;

        if (mass_density_g_per_cm3 <= 0.0) {
            throw std::runtime_error("Mass density must be positive");
        }

        if (atomic_mass_g_per_mol <= 0.0) {
            throw std::runtime_error("Atomic mass must be positive");
        }

        // Result: atoms / barn-cm.
        return mass_density_g_per_cm3 * avogadro / atomic_mass_g_per_mol * 1.0e-24;
    }

    inline Material make_material_from_ace(
        const AceFile& ace,
        double atom_density
    ) {
        Material material{
            .atom_density = atom_density,
            .mass_number = ace.header.atomic_weight_ratio,
            .micro_xs = extract_esz_table(ace)
        };

        material.validate();

        return material;
    }

    inline Material load_ace_material(
        const std::string& path,
        double atom_density
    ) {
        const AceFile ace = read_ascii_ace(path);
        return make_material_from_ace(ace, atom_density);
    }

    inline Material load_ace_material_from_mass_density(
        const std::string& path,
        double mass_density_g_per_cm3
    ) {
        const AceFile ace = read_ascii_ace(path);

        const double atom_density = atom_density_from_mass_density(
            mass_density_g_per_cm3,
            ace.header.atomic_weight_ratio
        );

        return make_material_from_ace(ace, atom_density);
    }

}
