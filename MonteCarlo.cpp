#include "MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
    int accepted = 0;

    while (accepted < config.trials) {
        double r_sim = dist_r(rng);
        double xi_sim = dist_xi(rng);

        // Controllo under-coupling: r > xi per differenziatori frazionari
        if (r_sim <= xi_sim || r_sim >= 1.0 || xi_sim <= 0.0) {
            res.redraws++; // <-- Corretto
            continue;
        }

        double neff_sim = dist_neff(rng);
        double ng_sim = std::max(1.5, dist_ng(rng));

        double df_sim = 0.0;
        if (config.enable_thermal_tuning) {
            df_sim = dist_df_tuned(rng);
        } else {
            double delta_neff = neff_sim - nominal_cascade.mode_index();
            df_sim = -f0 * (delta_neff / ng_sim);
        }

        MRRCascade perturbed = MRRCascade::perturbed(
            nominal_cascade, r_sim, xi_sim, neff_sim, ng_sim, df_sim);

        Simulation sim(perturbed, sim_samples);
        const auto &prop = sim.run(pulse, config.align_waveforms, false);

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
        accepted++;
    }

    double sum =
        std::accumulate(res.errors_Dn.begin(), res.errors_Dn.end(), 0.0);
    res.mean_error = sum / config.trials;

    double sq_sum = 0.0;
    for (double e : res.errors_Dn) {
        sq_sum += (e - res.mean_error) * (e - res.mean_error);
    }
    res.std_error = std::sqrt(sq_sum / config.trials);

    std::vector<double> sorted = res.errors_Dn;
    std::sort(sorted.begin(), sorted.end());
    res.median_error = sorted[config.trials / 2];
    res.max_error =
        sorted.back(); // <-- Corretto (max_error invece di worst_error)
    res.yield_rate =
        (static_cast<double>(passed_count) / config.trials) * 100.0;

    return res;
}

void MonteCarlo::Result::print_summary() const {
    std::printf("=====================================================\n");
    std::printf("           MONTE CARLO YIELD & ERROR REPORT          \n");
    std::printf("=====================================================\n");
    std::printf("Samples            : %zu\n", errors_Dn.size());
    std::printf("Yield (Dn <= 10%%)  : \033[1;32m%.2f %%\033[0m\n", yield_rate);
    std::printf("Mean Dn            : %.2f %%\n", mean_error);
    std::printf("Std deviation      : %.2f %%\n", std_error);
    std::printf("Median Dn          : %.2f %%\n", median_error);
    std::printf("Worst Dn           : %.2f %%\n", max_error); // <-- Corretto
    std::printf("Redrawn            : %ld (r <= xi)\n",
                redraws); // <-- Corretto (%ld)
    std::printf("=====================================================\n");
}
