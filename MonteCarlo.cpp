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

    std::random_device rd;
    std::mt19937_64 rng(rd());

    // Generatori gaussiani centrati sui valori nominali dell'anello
    std::normal_distribution<double> dist_r(nominal_ring.self_coupling(),
                                            config.sigma_r);
    std::normal_distribution<double> dist_xi(nominal_ring.round_trip_loss(),
                                             config.sigma_xi);
    std::normal_distribution<double> dist_neff(nominal_ring.mode_index(),
                                               config.sigma_neff);
    std::normal_distribution<double> dist_ng(nominal_ring.group_index(),
                                             config.sigma_ng);
    std::normal_distribution<double> dist_df_tuned(0.0, config.sigma_df_tuned);

    const double c = 2.99792458e8;
    const double f0 = c / config.lambda_0;

    int passed_count = 0;

    std::cout << "\n=== Inizio Simulazione Monte Carlo (" << config.trials
              << " iterazioni) ===" << std::endl;

    for (int i = 0; i < config.trials; ++i) {
        // Estrazione e vincoli fisici (0 < r < 1, 0 < xi < 1)
        double r_sim = std::clamp(dist_r(rng), 0.85, 0.9999);
        double xi_sim = std::clamp(dist_xi(rng), 0.85, 0.9999);
        double neff_sim = dist_neff(rng);
        double ng_sim = std::max(1.5, dist_ng(rng));

        // Calcolo del detuning df da variazione dell'indice di modo
        double df_sim = 0.0;
        if (config.enable_thermal_tuning) {
            // Se c'è controllo termico attivo, il detuning è limitato alla
            // precisione del riscaldatore
            df_sim = dist_df_tuned(rng);
        } else {
            // Detuning ottico puro: Delta f = - f0 * (Delta n_eff / n_g)
            double delta_neff = neff_sim - nominal_ring.mode_index();
            df_sim = -f0 * (delta_neff / ng_sim);
        }

        // Costruzione dell'anello perturbato
        MRR perturbed_ring(nominal_ring.radius(), r_sim, xi_sim, neff_sim,
                           ng_sim, df_sim);

        // Simulazione (senza Matplotlib/rendering grafico)
        Simulation sim(perturbed_ring, n, sim_samples);
        const auto &prop = sim.run(pulse, config.align_waveforms);

        double err_pct = prop.error_Dn * 100.0;
        res.errors_Dn.push_back(err_pct);
        res.r_samples.push_back(r_sim);
        res.xi_samples.push_back(xi_sim);
        res.neff_samples.push_back(neff_sim);
        res.ng_samples.push_back(ng_sim);
        res.df_samples.push_back(df_sim / 1e9);

        if (err_pct <= (config.yield_threshold * 100.0)) {
            passed_count++;
        }
    }

    // Statistiche descrittive
    double sum =
        std::accumulate(res.errors_Dn.begin(), res.errors_Dn.end(), 0.0);
    res.mean_error = sum / config.trials;

    double sq_sum = 0.0;
    for (double err : res.errors_Dn) {
        sq_sum += (err - res.mean_error) * (err - res.mean_error);
    }
    res.std_error = std::sqrt(sq_sum / config.trials);

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
    std::printf("Campioni totali : %zu\n", errors_Dn.size());
    std::printf("Resa (Dn <= 10%%): \033[1;32m%.2f %%\033[0m\n", yield_rate);
    std::printf("Errore Dn Medio : %.2f %%\n", mean_error);
    std::printf("Deviazione Std  : %.2f %%\n", std_error);
    std::printf("Mediana Dn      : %.2f %%\n", median_error);
    std::printf("Errore Massimo  : %.2f %%\n", max_error);
    std::printf("============================================\n\n");
}
