#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include <vector>

#include "Fabrication.hpp"
#include "MRR.hpp"
#include "MRRCascade.hpp"
#include "Simulation.hpp"

/**
 * @brief Monte Carlo yield and fabrication-tolerance study of the ring
 *        differentiator.
 */
class MonteCarlo {
  public:
    struct Config {
        int trials = 500;          // devices drawn
        double lambda_0 = 1550e-9; // carrier wavelength [m]

        /// Draw the geometry error once per device and derive r, xi, n_eff
        /// and n_g from it: one width bias sets both the gap and the mode
        /// index. False restores independent draws, for comparison only.
        bool correlated = true;

        /// Ring-to-ring correlation inside one cascade, [LU17]: one
        /// die-level error shared by every ring plus a small independent
        /// residual, rho = exp(-pitch/Lc). At the defaults rho ~ 0.99.
        double ring_pitch = fab::layout::ring_pitch;   // [m]
        double corr_length = fab::layout::corr_length; // [m]

        // Geometry, used when `correlated`.
        double sigma_width = fab::process::sigma_width;   // [m]
        double sigma_height = fab::process::sigma_height; // [m]
        double sigma_radius = fab::process::sigma_radius; // [m]
        double dxi_dwidth = fab::sensitivity::dxi_dwidth; // [1/m], unmeasured
        double dng_dwidth = fab::sensitivity::dng_dwidth; // [1/m]

        // Optical, used when not `correlated`.
        double sigma_r = fab::sigma_r();
        double sigma_xi = 0.002;
        double sigma_neff = fab::sigma_neff();
        double sigma_ng = 0.02;

        // Active thermal tuning. When on, a heater re-locks the carrier and
        // the neff-driven offset is replaced by the heater's residual error.
        bool enable_thermal_tuning = false;
        double sigma_df_tuned = 0.05e9; // residual lock error [Hz]

        double yield_threshold = 0.10; // pass if D_n <= 10%
        /// Plot-only: D_n is always measured on the overlapped waveforms.
        /// Nothing in the Monte Carlo draws, so this changes no result here.
        bool align_waveforms = false;
        // bool align_waveforms = true;

        unsigned long long seed = 1; // To reproduce results

        bool verbose = true; // progress line
    };

    struct Result {
        std::vector<double> errors_Dn; // measured D_n [%]
        std::vector<double> r_samples;
        std::vector<double> xi_samples;
        std::vector<double> neff_samples;
        std::vector<double> ng_samples;
        std::vector<double> df_samples; // [GHz]
        std::vector<double> n_samples;  // order actually realised, NaN if over-coupled
        // Geometry errors [m], empty on the independent path.
        std::vector<double> dwidth_samples;
        std::vector<double> dheight_samples;
        std::vector<double> dradius_samples;

        double mean_error = 0.0;
        double std_error = 0.0;
        double median_error = 0.0;
        double max_error = 0.0;
        double yield_rate = 0.0;      // percentage with D_n <= threshold
        double threshold_pct = 10.0;  // the threshold it was scored against [%]
        double mean_n = 0.0;     // achieved order, over-coupled devices excluded
        double std_n = 0.0;
        double rho_rings = 1.0;  // ring-to-ring correlation actually used
        size_t stages = 1;
        long over_coupled = 0; // devices with r <= xi, kept and scored

        void print_summary() const;
    };

    MonteCarlo(const MRRCascade &nominal_cascade,
               const Simulation::Input &pulse, const Config &config)
        : nominal_cascade(nominal_cascade), pulse(pulse), config(config) {}

    MonteCarlo(const MRRCascade &nominal_cascade,
               const Simulation::Input &pulse)
        : MonteCarlo(nominal_cascade, pulse, Config()) {}

    /// Draws and evaluates every device.
    Result run(long sim_samples = 65536) const;

  private:
    MRRCascade nominal_cascade; // the device as drawn
    Simulation::Input pulse;
    Config config;
};

#endif // MONTE_CARLO_HPP
