// Copyright 2026 Dawid Draguła
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mc_mini/geometry.hpp"

namespace mcm {

    struct SimulationConfig {
        std::uint64_t particle_count{100'000'000};
        double box_side_length{60.0};
        std::string material_file{"data/ace/C0.ACE"};
        double mass_density{2.26};

        [[nodiscard]] Box make_centered_box() const noexcept {
            const double half_length = 0.5 * box_side_length;

            return Box{
                .x_min = -half_length,
                .y_min = -half_length,
                .z_min = -half_length,
                .x_max = half_length,
                .y_max = half_length,
                .z_max = half_length
            };
        }
    };

    namespace config_detail {

        inline std::string read_file(const std::string& path) {
            std::ifstream file{path};

            if (!file) {
                throw std::runtime_error("Failed to open config file: " + path);
            }

            std::ostringstream stream;
            stream << file.rdbuf();

            return stream.str();
        }

        inline std::size_t value_position(
            const std::string& content,
            const std::string& key
        ) {
            const std::string quoted_key = "\"" + key + "\"";
            const std::size_t key_position = content.find(quoted_key);

            if (key_position == std::string::npos) {
                return std::string::npos;
            }

            const std::size_t colon_position = content.find(':', key_position + quoted_key.size());

            if (colon_position == std::string::npos) {
                throw std::runtime_error("Config key has no value: " + key);
            }

            std::size_t position = colon_position + 1;

            while (position < content.size() && std::isspace(static_cast<unsigned char>(content[position]))) {
                ++position;
            }

            return position;
        }

        inline std::uint64_t read_uint64(
            const std::string& content,
            const std::string& key,
            std::uint64_t fallback
        ) {
            const std::size_t position = value_position(content, key);

            if (position == std::string::npos) {
                return fallback;
            }

            char* end = nullptr;
            errno = 0;

            const long double value = std::strtold(content.c_str() + position, &end);

            if (
                end == content.c_str() + position ||
                errno == ERANGE ||
                !std::isfinite(value) ||
                value <= 0.0L ||
                value > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) ||
                value != std::trunc(value)
            ) {
                throw std::runtime_error("Config value must be a positive integer: " + key);
            }

            while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
                ++end;
            }

            if (*end != ',' && *end != '}' && *end != ']') {
                throw std::runtime_error("Config value has unexpected trailing characters: " + key);
            }

            return static_cast<std::uint64_t>(value);
        }

        inline double read_double(
            const std::string& content,
            const std::string& key,
            double fallback
        ) {
            const std::size_t position = value_position(content, key);

            if (position == std::string::npos) {
                return fallback;
            }

            char* end = nullptr;
            errno = 0;

            const double value = std::strtod(content.c_str() + position, &end);

            if (end == content.c_str() + position || errno == ERANGE) {
                throw std::runtime_error("Config value must be a number: " + key);
            }

            while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
                ++end;
            }

            if (*end != ',' && *end != '}' && *end != ']') {
                throw std::runtime_error("Config value has unexpected trailing characters: " + key);
            }

            return value;
        }

        inline std::string read_string(
            const std::string& content,
            const std::string& key,
            const std::string& fallback
        ) {
            const std::size_t position = value_position(content, key);

            if (position == std::string::npos) {
                return fallback;
            }

            if (position >= content.size() || content[position] != '"') {
                throw std::runtime_error("Config value must be a string: " + key);
            }

            std::string value;

            for (std::size_t i = position + 1; i < content.size(); ++i) {
                const char character = content[i];

                if (character == '"') {
                    return value;
                }

                if (character == '\\') {
                    if (i + 1 >= content.size()) {
                        throw std::runtime_error("Config string has incomplete escape: " + key);
                    }

                    value.push_back(content[++i]);
                } else {
                    value.push_back(character);
                }
            }

            throw std::runtime_error("Config string is not closed: " + key);
        }

    }

    inline void validate_config(const SimulationConfig& config) {
        if (config.particle_count == 0) {
            throw std::runtime_error("Config particle_count must be positive");
        }

        if (config.particle_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Config particle_count exceeds CUDA queue index range");
        }

        if (config.box_side_length <= 0.0) {
            throw std::runtime_error("Config box_side_length must be positive");
        }

        if (config.material_file.empty()) {
            throw std::runtime_error("Config material_file must not be empty");
        }

        if (config.mass_density <= 0.0) {
            throw std::runtime_error("Config mass_density must be positive");
        }
    }

    inline SimulationConfig load_simulation_config(const std::string& path = "config.json") {
        const std::string content = config_detail::read_file(path);

        SimulationConfig config{};
        config.particle_count = config_detail::read_uint64(
            content,
            "particle_count",
            config.particle_count
        );
        config.box_side_length = config_detail::read_double(
            content,
            "box_side_length",
            config.box_side_length
        );
        config.material_file = config_detail::read_string(
            content,
            "material_file",
            config.material_file
        );
        config.mass_density = config_detail::read_double(
            content,
            "mass_density",
            config.mass_density
        );

        validate_config(config);

        return config;
    }

    inline SimulationConfig load_simulation_config(int argc, char* argv[]) {
        if (argc > 1) {
            return load_simulation_config(argv[1]);
        }

        return load_simulation_config();
    }

}
