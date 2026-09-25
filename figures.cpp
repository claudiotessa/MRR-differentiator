// Every figure of docs/main.tex, written as PDF to docs/fig (or argv[1]).
// Run from the repository root: ./build/figures

#include <cstdio>
#include <string>
#include <vector>

#include "Fabrication.hpp"
#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"

// The reference device of [LIU25], as in main.cpp.
static const double R_ring = fab::paper::R;
static const double xi = fab::paper::xi;
static const double n_eff = fab::paper::n_eff;
static const double n_g = fab::paper::n_g;

static const double threshold_pct = fab::paper::D_accept * 100.0;
static const long N_fast = 32768; // converged, see Simulation::run()

static double error_pct(const MRRCascade &c, double T0) {
    Simulation sim(c, N_fast);
    return sim.run(Simulation::Input::gaussian(T0), false, false).error_Dn *
           100.0;
}

// The Monte Carlo of main.cpp: correlated geometry, one heater per ring.
static MonteCarlo::Config mc_config(int trials) {
    MonteCarlo::Config cfg;
    cfg.trials = trials;
    cfg.yield_threshold = fab::paper::D_accept;
    cfg.correlated = true;
    cfg.enable_thermal_tuning = true;
    cfg.sigma_df_tuned = fab::sigma_df_tuned;
    return cfg;
}

int main(int argc, char **argv) {
    Plotter::headless();
    Plotter::set_output_dir(argc > 1 ? argv[1] : "docs/fig");

    const double n = 0.54;
    const double T0 = fab::paper::input_T0; // 3 ps
    const MRRCascade design(n, R_ring, xi, n_eff, n_g);
    const double r_design = design.self_coupling();
    const Simulation::Input pulse = Simulation::Input::gaussian(T0);

    // --- Design ------------------------------------------------------
    std::printf(">>> Design figures\n");
    Plotter::plot_locus_map({0.1, 0.2, 0.3, 0.4, 0.54, 0.6, 0.7, 0.8, 0.9}, n,
                            r_design, xi);
    Plotter::save("locus-map");

    {
        // Fixed pulse against a pulse that scales with the ring, equal to
        // T0 at the design point: the second isolates the ring's shape.
        const double fwhm_design =
            MRR::fractional_order(n, R_ring, xi, n_eff, n_g).fwhm();
        std::vector<double> r, D_fixed, D_scaled;
        for (double x = 0.85; x <= 0.9951; x += 0.005) {
            const MRRCascade c(n, R_ring, x, n_eff, n_g);
            const double fwhm =
                MRR::fractional_order(n, R_ring, x, n_eff, n_g).fwhm();
            r.push_back(c.self_coupling());
            D_fixed.push_back(error_pct(c, T0));
            D_scaled.push_back(error_pct(c, T0 * fwhm_design / fwhm));
        }
        Plotter::plot_dn_along_locus(r, D_fixed, D_scaled, n, r_design);
        Plotter::save("dn-vs-locus");
    }

    // --- Nominal device ----------------------------------------------
    std::printf(">>> Nominal device\n");
    Plotter::plot_frequency_response(design);
    Plotter::save("freq-n054");
    Plotter::plot_frequency_response(MRRCascade(1.0, R_ring, xi, n_eff, n_g));
    Plotter::save("freq-n1");

    {
        Simulation sim(design);
        // Aligned: the power panel shows the pair D_n integrates.
        const auto &p = sim.run(pulse, true, false);
        Plotter::plot_time_power(p);
        Plotter::save("time-n054");
    }

    // --- Reproduction of [LIU25] -------------------------------------
    std::printf(">>> Error against the order\n");
    {
        // From 0.3: below it the ideal's t^-(n+1) tail does not converge in
        // any affordable window (n = 0.1 still moves 0.7 points at 4x).
        // Indexed, not accumulated, so n = 1 is exactly 1.
        std::vector<double> orders, D;
        for (int i = 0; i <= 36; ++i) {
            const double k = 0.30 + 0.05 * i;
            orders.push_back(k);
            Simulation sim(MRRCascade(k, R_ring, xi, n_eff, n_g));
            D.push_back(sim.run(pulse, false, false).error_Dn * 100.0);
        }
        const std::vector<double> bench_n = {0.54, 1.44, 2.10};
        const std::vector<double> bench_T0 = {fab::paper::input_T0,
                                              fab::paper::input_T0_144,
                                              fab::paper::input_T0_210};
        const std::vector<double> bench_paper = {
            fab::paper::D_054_fdtd * 100.0, fab::paper::D_144_fdtd * 100.0,
            fab::paper::D_210_fdtd * 100.0};
        std::vector<double> bench_ours;
        for (size_t i = 0; i < bench_n.size(); ++i)
            bench_ours.push_back(error_pct(
                MRRCascade(bench_n[i], R_ring, xi, n_eff, n_g), bench_T0[i]));
        Plotter::plot_dn_vs_order(orders, D, T0 * 1e12, bench_n, bench_ours,
                                  bench_paper, threshold_pct);
        Plotter::save("dn-vs-order");
    }

    // --- Robustness --------------------------------------------------
    std::printf(">>> Monte Carlo at the design point\n");
    const MonteCarlo::Result mc =
        MonteCarlo(design, pulse, mc_config(500)).run(N_fast);
    mc.print_summary();
    Plotter::plot_monte_carlo(mc, threshold_pct);
    Plotter::save("mc-hist");
    Plotter::plot_mc_scatter(mc, n, r_design, xi, threshold_pct);
    Plotter::save("mc-scatter");

    std::printf(">>> One tolerance at a time\n");
    {
        struct Only {
            std::string label;
            double w, h, R, df; // which sigmas stay on
            bool heater;
        };
        const double sw = fab::process::sigma_width,
                     sh = fab::process::sigma_height,
                     sR = fab::process::sigma_radius,
                     sdf = fab::sigma_df_tuned;
        const std::vector<Only> cases = {
            {"all", sw, sh, sR, sdf, true},
            {"width\n(gap, n_eff)", sw, 0, 0, 0, true},
            {"height\n(n_eff)", 0, sh, 0, 0, true},
            {"radius\n(xi)", 0, 0, sR, 0, true},
            {"heater\nresidual", 0, 0, 0, sdf, true},
            {"all,\nno heater", sw, sh, sR, 0, false}};

        std::vector<std::string> labels;
        std::vector<double> yield;
        for (const Only &c : cases) {
            MonteCarlo::Config cfg = mc_config(300);
            cfg.sigma_width = c.w;
            cfg.sigma_height = c.h;
            cfg.sigma_radius = c.R;
            cfg.sigma_df_tuned = c.df;
            cfg.enable_thermal_tuning = c.heater;
            labels.push_back(c.label);
            yield.push_back(MonteCarlo(design, pulse, cfg).run(N_fast).yield_rate);
        }
        Plotter::plot_tolerance_breakdown(labels, yield, threshold_pct);
        Plotter::save("mc-breakdown");
    }

    std::printf(">>> Yield along the design curve\n");
    {
        // The same tolerances at every point: a different xi is a different
        // radius or process, whose sensitivities we do not have.
        std::vector<double> r, D_nom, yield;
        for (double x : {0.88, 0.90, 0.92, 0.93, xi, 0.95, 0.96, 0.97, 0.98}) {
            const MRRCascade c(n, R_ring, x, n_eff, n_g);
            r.push_back(c.self_coupling());
            D_nom.push_back(error_pct(c, T0));
            yield.push_back(MonteCarlo(c, pulse, mc_config(300)).run(N_fast).yield_rate);
        }
        Plotter::plot_yield_along_locus(r, D_nom, yield, n, r_design,
                                        threshold_pct);
        Plotter::save("mc-yield");
    }

    return 0;
}
