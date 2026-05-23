#include <chrono>
#include <iostream>

namespace mcm::timer {
    using Clock = std::chrono::steady_clock;

    struct TimeStats {
        double total_time = 0.0;
        double initialization_time = 0.0;
        double transport_time = 0.0;
    } stats;

    struct TimeRecords {
        Clock::time_point program_start;
        Clock::time_point initialization_end;
        Clock::time_point transport_end;
    } records;

    [[nodiscard]] double seconds_between(
        Clock::time_point start,
        Clock::time_point end
    ) {
        return std::chrono::duration<double>(end - start).count();
    }

    void print_timing_results(std::uint64_t particle_count) {
        std::cout << "==== TIMING RESULTS =====" << std::endl << std::endl;
        std::cout << "Total simulation time: " << stats.total_time << " s\n";
        std::cout << "Initialization time: " << stats.initialization_time << " s\n";
        std::cout << "Transport time: " << stats.transport_time << " s\n";
        std::cout << "Particle histories per second: "
            << static_cast<double>(particle_count) / stats.transport_time
            << std::endl;
    }

    void record_start() noexcept {
        records.program_start = Clock::now();
    }

    void record_initialization_end() noexcept {
        records.initialization_end = Clock::now();
        stats.initialization_time = seconds_between(records.program_start, records.initialization_end);
    }

    void record_transport_end() noexcept {
        records.transport_end = Clock::now();
        stats.transport_time = seconds_between(records.initialization_end, records.transport_end);
        stats.total_time = seconds_between(records.program_start, records.transport_end);
    }
}