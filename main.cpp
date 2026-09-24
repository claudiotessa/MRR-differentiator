#include "MRR.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"
#include <iostream>

// Waveguide parameters shared by every figure.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // mode index, 220 nm SOI strip
static const double n_g = 4.2;       // group index, same waveguide

// int main() {
//     const double n = 0.54; // fractional order
//
//     MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff, n_g);
//
//     // The paper drives its 0.54-order device with a Gaussian sized against
//     the
//     // ring's Eq. (4) width. Other shapes are available - super_gaussian(),
//     // sech(), rectangular() - but the Gaussian is the one the paper's error
//     // figures are measured with, so it is the only fair comparison.
//     Simulation sim(ring, n);
//
//     // Create the impulse coupled to the ring
//     Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);
//
//     sim.run(pulse, true);
//     Plotter::plot_all(sim);
//
//     return 0;
// }

int main() {
    const double n = 0.54; // target fractional order

    // 1. The nominal device, as drawn
    MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff, n_g);
    Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);

    // 2. Check the unperturbed case
    std::cout << ">>> Nominal case..." << std::endl;
    Simulation nominal_sim(ring, n);
    nominal_sim.run(pulse, false); // plot the ideal unshifted
    std::cout << "Nominal Dn: " << nominal_sim.get_error() * 100.0
              << " %\n"
              << std::endl;

    // 3. Monte Carlo yield analysis
    MonteCarlo::Config config;
    config.trials = 500;           // chips simulated
    config.yield_threshold = 0.10; // pass if Dn <= 10%
    config.align_waveforms = false; // plot-only flag; no effect on D_n

    // Fabrication tolerances typical of 220 nm SOI
    config.sigma_r = 0.0015;    // coupling, from the lithographic gap
    config.sigma_xi = 0.0020;   // propagation loss
    config.sigma_neff = 1.0e-4; // geometry error on neff
    config.sigma_ng = 0.02;     // group index

    // Thermal tuning: untuned, a sigma_neff of 1e-4 shifts the resonance by
    // ~4 GHz and the yield collapses to ~0%. Tuned, a microheater re-locks the
    // carrier and only its residual error is left.
    config.enable_thermal_tuning = true;
    config.sigma_df_tuned = 0.05e9; // residual thermal lock error: 50 MHz

    // 4. Run
    MonteCarlo mc(ring, n, pulse, config);
    MonteCarlo::Result results =
        mc.run(32768); // power of two: kissfft is slowest on odd factors

    // 5. Report
    results.print_summary();

    // 6. Figures, straight to fig/ under the names the slides expect
    Plotter::plot_time_domain(nominal_sim.get_last_result());
    Plotter::save("time-n054");
    Plotter::plot_frequency_response(ring, n);
    Plotter::save("freq-n054");
    // TODO: mc-scatter, mc-hist from `results`

    return 0;
}
