#include "Fabrication.hpp"
#include "MRR.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"
#include <cstdio>
#include <iostream>

int main() {
    const double n = 0.54; // target fractional order

    // 1. The paper's own device: the tolerances in Fabrication.hpp are tied
    //    to this geometry, so using anything else invalidates them.
    MRR ring = MRR::fractional_order(n, fab::paper::R, fab::paper::xi,
                                     fab::paper::n_eff, fab::paper::n_g);

    // Their input too.
    Simulation::Input pulse = Simulation::Input::gaussian(fab::paper::input_T0);

    // 2. Against the published error
    std::cout << ">>> Nominal case..." << std::endl;
    Simulation nominal_sim(ring, n);
    nominal_sim.run(pulse, false); // plot the ideal unshifted
    std::printf("r solved   : %.4f   (paper: %.4f)\n", ring.self_coupling(),
                fab::paper::r_n054);
    std::printf("Nominal Dn : %.2f %%  (paper: %.1f %% theory, %.2f %% FDTD)\n\n",
                nominal_sim.get_error() * 100.0,
                fab::paper::D_054_theory * 100.0,
                fab::paper::D_054_fdtd * 100.0);

    // 3. Monte Carlo yield analysis, tolerances from Fabrication.hpp
    MonteCarlo::Config config;
    config.trials = 500;
    config.yield_threshold = fab::paper::D_accept; // the paper's own bar
    config.align_waveforms = false; // plot-only flag; no effect on D_n

    // The defaults already come from Fabrication.hpp; this is the record.
    std::printf("sigma_w    : %.3f nm   [LU17]\n"
                "sigma_h    : %.3f nm   [LU17]\n"
                "sigma_R    : %.3f nm   (assumed = sigma_w)\n"
                "-> sigma_r : %.2e     via dr/dgap  = 8.2e-4 /nm  [LIU25]\n"
                "-> sigma_xi: %.2e     via dxi/dR   = 2.9e-4 /nm  [LIU25]\n"
                "-> sigma_ne: %.2e     via dneff/dw = 3.0e-3 /nm  [ROB22]\n"
                "untuned df : %.1f GHz scatter vs FSR = %.1f GHz\n\n",
                config.sigma_width * 1e9, config.sigma_height * 1e9,
                config.sigma_radius * 1e9, fab::sigma_r(),
                fab::sensitivity::dxi_dradius * config.sigma_radius,
                fab::sigma_neff(),
                fab::sigma_df_untuned(fab::paper::n_g) / 1e9,
                ring.fsr() / 1e9);

    config.enable_thermal_tuning = true;
    config.sigma_df_tuned = fab::sigma_df_tuned;

    // 4. Run. Set config.correlated = false to compare against independent
    //    draws; Fabrication.hpp records what that comparison gives.
    MonteCarlo::Result results = MonteCarlo(ring, n, pulse, config).run(32768);
    results.print_summary();
    (void)results; // TODO: mc-scatter, mc-hist

    // 5. Figures, straight to fig/ under the names the slides expect
    Plotter::plot_time_domain(nominal_sim.get_last_result());
    Plotter::save("time-n054");
    Plotter::plot_frequency_response(ring, n);
    Plotter::save("freq-n054");

    return 0;
}
