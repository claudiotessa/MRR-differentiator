#include "MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>

MonteCarlo::Result MonteCarlo::run(long sim_samples) const {
    Result res;
    res.errors_Dn.reserve(config.trials);
    res.r_samples.reserve(config.trials);
    res.xi_samples.reserve(config.trials);
    res.neff_samples.reserve(config.trials);
    res.ng_samples.reserve(config.trials);
    res.df_samples.reserve(config.trials);
    res.n_samples.reserve(config.trials);
    res.dwidth_samples.reserve(config.trials);
    res.dheight_samples.reserve(config.trials);
    res.dradius_samples.reserve(config.trials);

    std::random_device rd;
    std::mt19937_64 rng(rd());

    // Errori geometrici (a media zero)
    std::normal_distribution<double> dist_w(0.0, config.sigma_width);
    std::normal_distribution<double> dist_h(0.0, config.sigma_height);
    std::normal_distribution<double> dist_R(0.0, config.sigma_radius);

    // Parametri ottici centrati sulla cascata nominale
    std::normal_distribution<double> dist_r(nominal_cascade.self_coupling(),
                                            config.sigma_r);
    std::normal_distribution<double> dist_xi(nominal_cascade.round_trip_loss(),
                                             config.sigma_xi);
    std::normal_distribution<double> dist_neff(nominal_cascade.mode_index(),
                                               config.sigma_neff);
    std::normal_distribution<double> dist_ng(nominal_cascade.group_index(),
                                             config.sigma_ng);
    std::normal_distribution<double> dist_df_tuned(0.0, config.sigma_df_tuned);

    const double c = 2.99792458e8;
    const double f0 = c / config.lambda_0;

    int passed_count = 0;

    if (config.verbose) {
        std::cout << "\n=== Monte Carlo (" << config.trials << " trials, "
                  << (config.correlated ? "correlated geometry" : "independent")
                  << ") ===" << std::endl;
    }

    for (int i = 0; i < config.trials; ++i) {
        double r_sim = 0.0, xi_sim = 0.0, neff_sim = 0.0, ng_sim = 0.0;
        double R_sim = nominal_cascade.radius();
        double dw = 0.0, dh = 0.0, dR = 0.0;

        for (int attempt = 0;; ++attempt) {
            if (config.correlated) {
                dw = dist_w(rng);
                dh = dist_h(rng);
                dR = dist_R(rng);

                // Allargando la guida (+dw) il gap si riduce, aumentando
                // l'accoppiamento (r cala)
                r_sim = nominal_cascade.self_coupling() -
                        fab::sensitivity::dr_dgap * dw;
                neff_sim = nominal_cascade.mode_index() +
                           fab::sensitivity::dneff_dwidth * dw +
                           fab::sensitivity::dneff_dheight * dh;
                xi_sim = nominal_cascade.round_trip_loss() +
                         fab::sensitivity::dxi_dradius * dR +
                         config.dxi_dwidth * dw;
                ng_sim = nominal_cascade.group_index() + config.dng_dwidth * dw;
                R_sim = nominal_cascade.radius() + dR;
            } else {
                r_sim = dist_r(rng);
                xi_sim = dist_xi(rng);
                neff_sim = dist_neff(rng);
                ng_sim = dist_ng(rng);
                R_sim = nominal_cascade.radius();
            }

            r_sim = std::clamp(r_sim, 0.5, 0.9999);
            xi_sim = std::clamp(xi_sim, 0.5, 0.9999);
            ng_sim = std::max(1.5, ng_sim);

            if (!config.enforce_under_coupled || r_sim > xi_sim)
                break;

            ++res.redraws;
            if (attempt >= 999) {
                throw std::runtime_error(
                    "MonteCarlo: cannot draw r > xi - the coupling margin is "
                    "too small for these tolerances");
            }
        }

        // Calcolo del detuning da disallineamento frequenziale
        double df_sim = 0.0;
        if (config.enable_thermal_tuning) {
            df_sim = dist_df_tuned(rng);
        } else {
            double delta_neff = neff_sim - nominal_cascade.mode_index();
            df_sim = -f0 * (delta_neff / ng_sim);
        }

        // Istanziazione della cascata perturbata (incluso il raggio perturbato
        // R_sim)
        MRRCascade perturbed = MRRCascade::perturbed(
            nominal_cascade, r_sim, xi_sim, neff_sim, ng_sim, df_sim, R_sim);

        // Simulazione numerica silente
        Simulation sim(perturbed, sim_samples);
        const auto &prop = sim.run(pulse, config.align_waveforms, false);

        double err_pct = prop.error_Dn * 100.0;
        res.errors_Dn.push_back(err_pct);
        res.r_samples.push_back(r_sim);
        res.xi_samples.push_back(xi_sim);
        res.neff_samples.push_back(neff_sim);
        res.ng_samples.push_back(ng_sim);
        res.df_samples.push_back(df_sim / 1e9);
        res.n_samples.push_back(perturbed.achieved_order());

        if (config.correlated) {
            res.dwidth_samples.push_back(dw);
            res.dheight_samples.push_back(dh);
            res.dradius_samples.push_back(dR);
        }

        if (err_pct <= (config.yield_threshold * 100.0)) {
            passed_count++;
        }
    }

    // Statistiche sull'errore Dn
    double sum =
        std::accumulate(res.errors_Dn.begin(), res.errors_Dn.end(), 0.0);
    res.mean_error = sum / config.trials;

    double sq_sum = 0.0;
    for (double e : res.errors_Dn) {
        sq_sum += (e - res.mean_error) * (e - res.mean_error);
    }
    res.std_error = std::sqrt(sq_sum / config.trials);

    // Statistiche sull'ordine di derivazione effettivamente conseguito
    double n_sum =
        std::accumulate(res.n_samples.begin(), res.n_samples.end(), 0.0);
    res.mean_n = n_sum / config.trials;
    double n_sq = 0.0;
    for (double v : res.n_samples) {
        n_sq += (v - res.mean_n) * (v - res.mean_n);
    }
    res.std_n = std::sqrt(n_sq / config.trials);

    std::vector<double> sorted_err = res.errors_Dn;
    std::sort(sorted_err.begin(), sorted_err.end());
    res.median_error = sorted_err[config.trials / 2];
    res.max_error = sorted_err.back();
    res.yield_rate =
        (static_cast<double>(passed_count) / config.trials) * 100.0;

    return res;
}

void MonteCarlo::Result::print_summary() const {
    std::printf("\n============================================\n");
    std::printf("      MONTE CARLO YIELD & ERROR REPORT       \n");
    std::printf("============================================\n");
    std::printf("Samples       : %zu\n", errors_Dn.size());
    std::printf("Yield (Dn<=10%%): \033[1;32m%.2f %%\033[0m\n", yield_rate);
    std::printf("Mean Dn       : %.2f %%\n", mean_error);
    std::printf("Std deviation : %.2f %%\n", std_error);
    std::printf("Median Dn     : %.2f %%\n", median_error);
    std::printf("Worst Dn      : %.2f %%\n", max_error);
    std::printf("Achieved n    : %.4f +/- %.4f\n", mean_n, std_n);
    if (redraws > 0) {
        std::printf("Redrawn       : %ld (r <= xi)\n", redraws);
    }
    std::printf("============================================\n\n");
}
