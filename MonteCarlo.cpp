#include "MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

/// One-line completion bar, redrawn in place. Only every percent, so a long
/// run does not spend its time writing to the terminal.
void progress_bar(int done, int total) {
    const int width = 32;
    if (done < total && total > width &&
        done % std::max(1, total / 100) != 0)
        return;

    const double frac = static_cast<double>(done) / total;
    const int filled = static_cast<int>(frac * width);
    std::printf("\r  [");
    for (int i = 0; i < width; ++i)
        std::putchar(i < filled ? '#' : '.');
    std::printf("] %3.0f%%  %d/%d", frac * 100.0, done, total);
    std::fflush(stdout);
}

} // namespace

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

    std::mt19937_64 rng(config.seed);

    // Tutte le estrazioni passano da una normale standard: le sigma entrano
    // dopo, insieme alla correlazione anello-anello.
    std::normal_distribution<double> zn(0.0, 1.0);

    // [LU17]: gli anelli di una cascata distano decine di um, ben dentro la
    // lunghezza di correlazione millimetrica, quindi condividono quasi tutto
    // l'errore geometrico. x_i = sigma * (sqrt(rho)*z_comune + sqrt(1-rho)*z_i)
    const size_t S = std::max<size_t>(1, nominal_cascade.num_stages());
    const double rho =
        std::clamp(fab::layout::rho(config.ring_pitch, config.corr_length),
                   0.0, 1.0);
    const double w_com = std::sqrt(rho);       // peso della quota comune
    const double w_ind = std::sqrt(1.0 - rho); // peso del residuo per anello

    const double c = 2.99792458e8;
    const double f0 = c / config.lambda_0;

    const double r_nom = nominal_cascade.self_coupling();
    const double xi_nom = nominal_cascade.round_trip_loss();
    const double neff_nom = nominal_cascade.mode_index();
    const double ng_nom = nominal_cascade.group_index();
    const double R_nom = nominal_cascade.radius();

    res.rho_rings = rho;
    res.stages = S;

    int passed_count = 0;

    if (config.verbose) {
        std::printf("\n=== Monte Carlo: %d trials, %zu ring%s, %s, "
                    "rho(ring-ring) = %.4f ===\n",
                    config.trials, S, S == 1 ? "" : "s",
                    config.correlated ? "correlated geometry" : "independent",
                    rho);
    }

    std::vector<MRRCascade::StageParams> sp(S);

    for (int i = 0; i < config.trials; ++i) {
        double dw_avg = 0.0, dh_avg = 0.0, dR_avg = 0.0;

        for (int attempt = 0;; ++attempt) {
            // Quota comune a tutto il chip, estratta una volta per device.
            const double zw = zn(rng), zh = zn(rng), zR = zn(rng);
            const double zr = zn(rng), zx = zn(rng), zne = zn(rng),
                         zng = zn(rng);
            auto shared = [&](double z) { return w_com * z + w_ind * zn(rng); };

            bool ok = true;
            dw_avg = dh_avg = dR_avg = 0.0;

            for (size_t k = 0; k < S; ++k) {
                MRRCascade::StageParams &p = sp[k];
                double dw = 0.0, dh = 0.0, dR = 0.0;

                if (config.correlated) {
                    dw = config.sigma_width * shared(zw);
                    dh = config.sigma_height * shared(zh);
                    dR = config.sigma_radius * shared(zR);

                    // Allargando la guida (+dw) il gap si chiude e
                    // l'accoppiamento cresce: r cala.
                    p.r = r_nom - fab::sensitivity::dr_dgap * dw;
                    p.n_eff = neff_nom + fab::sensitivity::dneff_dwidth * dw +
                              fab::sensitivity::dneff_dheight * dh;
                    p.xi = xi_nom + fab::sensitivity::dxi_dradius * dR +
                           config.dxi_dwidth * dw;
                    p.n_g = ng_nom + config.dng_dwidth * dw;
                    p.R = R_nom + dR;
                } else {
                    p.r = r_nom + config.sigma_r * shared(zr);
                    p.xi = xi_nom + config.sigma_xi * shared(zx);
                    p.n_eff = neff_nom + config.sigma_neff * shared(zne);
                    p.n_g = ng_nom + config.sigma_ng * shared(zng);
                    p.R = R_nom;
                }

                p.r = std::clamp(p.r, 0.5, 0.9999);
                p.xi = std::clamp(p.xi, 0.5, 0.9999);
                p.n_g = std::max(1.5, p.n_g);

                // Ogni anello ha il suo heater, quindi il residuo di aggancio
                // e' indipendente; senza heater il detuning segue n_eff ed e'
                // correlato quanto la geometria.
                p.df = config.enable_thermal_tuning
                           ? config.sigma_df_tuned * zn(rng)
                           : -f0 * (p.n_eff - neff_nom) / p.n_g;

                dw_avg += dw;
                dh_avg += dh;
                dR_avg += dR;

                if (config.enforce_under_coupled && !(p.r > p.xi))
                    ok = false;
            }

            dw_avg /= static_cast<double>(S);
            dh_avg /= static_cast<double>(S);
            dR_avg /= static_cast<double>(S);

            if (ok)
                break;

            ++res.redraws;
            if (attempt >= 999) {
                throw std::runtime_error(
                    "MonteCarlo: cannot draw r > xi - the coupling margin is "
                    "too small for these tolerances");
            }
        }

        MRRCascade perturbed = MRRCascade::perturbed(nominal_cascade, sp);

        Simulation sim(perturbed, sim_samples);
        const auto &prop = sim.run(pulse, config.align_waveforms, false);

        double err_pct = prop.error_Dn * 100.0;
        res.errors_Dn.push_back(err_pct);
        res.r_samples.push_back(perturbed.self_coupling());
        res.xi_samples.push_back(perturbed.round_trip_loss());
        res.neff_samples.push_back(perturbed.mode_index());
        res.ng_samples.push_back(perturbed.group_index());
        res.df_samples.push_back(perturbed.resonance_offset() / 1e9);
        res.n_samples.push_back(perturbed.achieved_order());

        if (config.correlated) {
            res.dwidth_samples.push_back(dw_avg);
            res.dheight_samples.push_back(dh_avg);
            res.dradius_samples.push_back(dR_avg);
        }

        if (err_pct <= (config.yield_threshold * 100.0)) {
            passed_count++;
        }

        if (config.verbose)
            progress_bar(i + 1, config.trials);
    }
    if (config.verbose)
        std::printf("\n");

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
    if (stages > 1) {
        std::printf("Rings         : %zu, rho(ring-ring) = %.4f\n", stages,
                    rho_rings);
    }
    if (redraws > 0) {
        std::printf("Redrawn       : %ld (r <= xi)\n", redraws);
    }
    std::printf("============================================\n\n");
}
