#pragma once

namespace mcm {
    
    struct Material {
        double xs_a{};
        double xs_s{};

        [[nodiscard]] double xs_t() const noexcept {
            return xs_a + xs_s;
        }

        [[nodiscard]] double absorption_probability() const noexcept {
            return xs_a / xs_t();
        }

        [[nodiscard]] double scattering_probability() const noexcept {
            return xs_s / xs_t();
        }
    };

}
