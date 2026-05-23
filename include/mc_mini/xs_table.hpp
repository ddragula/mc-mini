#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcm {

    struct XsValues {
        double total{};
        double absorption{};
        double elastic{};
        double heating{};

        [[nodiscard]] double supported_total() const noexcept {
            return absorption + elastic;
        }
    };

    struct XsTable {
        // ACE ESZ block:
        // energies[i]    -> energy grid, MeV
        // total[i]       -> total microscopic XS, barn
        // absorption[i]  -> absorption microscopic XS, barn
        // elastic[i]     -> elastic microscopic XS, barn
        // heating[i]     -> heating number
        std::vector<double> energies;
        std::vector<double> total;
        std::vector<double> absorption;
        std::vector<double> elastic;
        std::vector<double> heating;

        void validate(const std::string& name = "XsTable") const {
            const std::size_t size = energies.size();

            if (size == 0) {
                throw std::runtime_error(name + ": empty energy grid");
            }

            if (
                total.size() != size ||
                absorption.size() != size ||
                elastic.size() != size ||
                heating.size() != size
            ) {
                throw std::runtime_error(name + ": vector size mismatch");
            }

            for (std::size_t i = 1; i < size; ++i) {
                if (energies[i] <= energies[i - 1]) {
                    throw std::runtime_error(name + ": energies must be strictly increasing");
                }
            }

            for (std::size_t i = 0; i < size; ++i) {
                if (total[i] < 0.0 || absorption[i] < 0.0 || elastic[i] < 0.0) {
                    throw std::runtime_error(name + ": cross sections must be non-negative");
                }
            }
        }

        [[nodiscard]] XsValues value_at(double energy) const {
            if (energies.empty()) {
                throw std::runtime_error("XsTable: empty energy grid");
            }

            if (energy <= energies.front()) {
                return values_at(0);
            }

            if (energy >= energies.back()) {
                return values_at(energies.size() - 1);
            }

            const auto upper = std::upper_bound(
                energies.begin(),
                energies.end(),
                energy
            );

            const std::size_t right = static_cast<std::size_t>(
                upper - energies.begin()
            );

            const std::size_t left = right - 1;

            const double e0 = energies[left];
            const double e1 = energies[right];
            const double t = (energy - e0) / (e1 - e0);

            return XsValues{
                .total = interpolate(total[left], total[right], t),
                .absorption = interpolate(absorption[left], absorption[right], t),
                .elastic = interpolate(elastic[left], elastic[right], t),
                .heating = interpolate(heating[left], heating[right], t)
            };
        }

    private:
        [[nodiscard]] XsValues values_at(std::size_t index) const {
            return XsValues{
                .total = total[index],
                .absorption = absorption[index],
                .elastic = elastic[index],
                .heating = heating[index]
            };
        }

        [[nodiscard]] static double interpolate(
            double left,
            double right,
            double t
        ) noexcept {
            return left + t * (right - left);
        }
    };

}
