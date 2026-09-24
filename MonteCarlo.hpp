#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include <vector>

#include "MRR.hpp"
#include "Simulation.hpp"

/**
 * @brief Monte Carlo yield and fabrication-tolerance study of the ring
 *        differentiator.
 */
class MonteCarlo {
  public:
    struct Config {
        int trials = 500; // devices drawn
        double lambda_0 =
            1550e-9; // carrier wavelength [m]

        // Standard deviations, Gaussian about the nominal values.
        double sigma_r =
            0.0015; // coupler gap (self-coupling)
        double sigma_xi = 0.002;  // sidewall roughness / loss
        double sigma_neff = 2e-4; // geometry error on the mode index
        double sigma_ng = 0.02;   // group index

        // Active thermal tuning. When on, a heater re-locks the carrier and
        // the neff-driven offset is replaced by the heater's residual error.
        bool enable_thermal_tuning = false;
        double sigma_df_tuned =
            0.05e9; // residual lock error [Hz]

        double yield_threshold = 0.10; // pass if D_n <= 10%
        bool align_waveforms = true;   // measure shape error only
    };

    struct Result {
        std::vector<double> errors_Dn; // measured D_n [%]
        std::vector<double> r_samples;
        std::vector<double> xi_samples;
        std::vector<double> neff_samples;
        std::vector<double> ng_samples;
        std::vector<double> df_samples; // [GHz]

        double mean_error = 0.0;
        double std_error = 0.0;
        double median_error = 0.0;
        double max_error = 0.0;
        double yield_rate = 0.0; // percentage with D_n <= threshold

        void print_summary() const;
    };

    // Default configuration
    MonteCarlo(const MRR &nominal_ring, double n,
               const Simulation::Input &pulse)
        : MonteCarlo(nominal_ring, n, pulse, Config()) {}

    // Custom configuration
    MonteCarlo(const MRR &nominal_ring, double n,
               const Simulation::Input &pulse, const Config &config)
        : nominal_ring(nominal_ring), n(n), pulse(pulse), config(config) {}

    /// Draws and evaluates every device.
    Result run(long sim_samples = 50000) const;

  private:
    MRR nominal_ring; // the device as drawn
    double n;
    Simulation::Input pulse;
    Config config;
};

#endif // MONTE_CARLO_HPP
