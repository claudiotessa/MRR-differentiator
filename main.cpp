#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Fabrication.hpp"
#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"

// Reference device: the theory-guided design of [LIU25] Sec. 2, via
// Fabrication.hpp. The tolerances were derived on this geometry, so changing
// it invalidates them.
static const double R_ring = fab::paper::R;    // 1.9 um
static const double xi = fab::paper::xi;       // 0.9428, set by bend loss
static const double n_eff = fab::paper::n_eff; // 2.25
static const double n_g = fab::paper::n_g;     // 4.05, Lumerical

int main() {
    // =========================================================================
    // 1. BENCHMARK ON THE PAPER'S ORDERS
    // =========================================================================
    // Each case is driven with the pulse [LIU25] actually used, not one
    // matched to the cascade: the input is part of what is being reproduced.
    struct Case {
        double n;
        double T0;      // input half-width [s], as stated by [LIU25]
        double D_paper; // the error [LIU25] reports
    };
    const std::vector<Case> paper_cases = {
        {0.54, fab::paper::input_T0, fab::paper::D_054_fdtd},
        {1.44, fab::paper::input_T0_144, fab::paper::D_144_fdtd},
        {2.10, fab::paper::input_T0_210, fab::paper::D_210_fdtd}};

    std::cout << "================================================="
                 "==================\n"
                 "              BENCHMARK AGAINST THE CASES OF [LIU25]"
                 "\n"
                 "================================================="
                 "==================\n";
    std::cout << std::left << std::setw(9) << "Order n" << std::setw(7)
              << "Rings" << std::setw(11) << "n per ring" << std::setw(12)
              << "Band [GHz]" << std::setw(11) << "T0 [ps]" << std::setw(11)
              << "D_n [%]" << std::setw(11) << "[LIU25]" << "\n";
    std::cout << "-------------------------------------------------"
                 "------------------\n";

    for (const Case &c : paper_cases) {
        MRRCascade casc(c.n, R_ring, xi, n_eff, n_g);
        Simulation::Input in = Simulation::Input::gaussian(c.T0);

        Simulation sim(casc);
        const auto &res = sim.run(in, false, false); // silent

        std::cout << std::left << std::fixed << std::setprecision(2)
                  << std::setw(9) << c.n << std::setw(7) << casc.num_stages()
                  << std::setw(11) << casc.stage_order() << std::setprecision(1)
                  << std::setw(12) << casc.usable_band() / 1e9 << std::setw(11)
                  << c.T0 * 1e12 << std::setprecision(2) << std::setw(11)
                  << res.error_Dn * 100.0 << std::setw(11) << c.D_paper * 100.0
                  << "\n";
    }
    std::cout << "================================================="
                 "==================\n";
    std::cout << "Driven with the pulse [LIU25] states for each case.\n"
                 "[LIU25] column is their FDTD result.\n"
                 "Their theory-only figure for n = 0.54 is "
              << std::setprecision(1) << fab::paper::D_054_theory * 100.0
              << " %.\n\n";

    // =========================================================================
    // 2. DETAILED NOMINAL SIMULATION
    // =========================================================================
    const double target_n = 1.44;
    std::cout << ">>> Nominal simulation for n = " << std::setprecision(2)
              << target_n << " ...\n";

    MRRCascade cascade(target_n, R_ring, xi, n_eff, n_g);
    // The paper's own input for this device: 3 ps for a single ring (Sec. 2),
    // 7 ps once it is a cascade (Sec. 3.B). Keyed off the order so that
    // retargeting target_n does not silently leave the wrong pulse behind.
    Simulation::Input pulse = Simulation::Input::gaussian(
        target_n <= 1.0   ? fab::paper::input_T0
        : target_n < 2.0  ? fab::paper::input_T0_144
                          : fab::paper::input_T0_210);

    Simulation nominal_sim(cascade);
    nominal_sim.print_setup(pulse);

    // align = false: the plots show the waveforms where they really fall.
    // D_n is measured on the aligned pair regardless.
    const auto &nominal_res = nominal_sim.run(pulse, false, false);
    std::cout << "Nominal result: " << nominal_res.caption << "\n\n";

    // =========================================================================
    // 3. MONTE CARLO TOLERANCE ANALYSIS OF THE DESIGNED DEVICE
    // =========================================================================
    std::cout << ">>> Starting Monte Carlo tolerance analysis..." << std::endl;

    MonteCarlo::Config mc_cfg;
    mc_cfg.trials = 500;
    mc_cfg.yield_threshold = fab::paper::D_accept; // the 10% bar of [LIU25]
    mc_cfg.align_waveforms = false;

    // One geometry error per chip; r, xi, n_eff and n_g all follow from it.
    // Sigmas and sensitivities come from Fabrication.hpp.
    mc_cfg.correlated = true;

    // A per-ring heater re-locks the resonance to the laser.
    mc_cfg.enable_thermal_tuning = true;
    mc_cfg.sigma_df_tuned = fab::sigma_df_tuned;

    std::printf("Tolerances (Fabrication.hpp): sigma_w = %.3f nm [LU17], "
                "sigma_h = %.3f nm [LU17]\n"
                "  -> sigma_r = %.2e, sigma_neff = %.2e, "
                "rho(ring-ring) = %.4f\n",
                fab::process::sigma_width * 1e9,
                fab::process::sigma_height * 1e9, fab::sigma_r(),
                fab::sigma_neff(), fab::layout::rho());

    MonteCarlo mc(cascade, pulse, mc_cfg);
    // FFT length: a power of two, for Eigen's radix-2 path.
    MonteCarlo::Result mc_res = mc.run(32768);

    mc_res.print_summary();

    // =========================================================================
    // 4. FIGURES
    // =========================================================================
    std::cout << ">>> Opening figures (close the windows to finish)..."
              << std::endl;
    Plotter::plot_all(nominal_sim, false); // waveforms and dB/phase spectrum
    Plotter::plot_monte_carlo(mc_res, mc_cfg.yield_threshold * 100.0);
    Plotter::show();

    return 0;
}
