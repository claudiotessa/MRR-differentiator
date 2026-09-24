#include "MRRCascade.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

MRRCascade::MRRCascade(double n, double R, double xi, double n_eff, double n_g,
                       double df)
    : total_order(n), R(R), xi(xi), n_eff(n_eff), n_g(n_g), df(df) {
    if (n <= 0.0) {
        throw std::invalid_argument(
            "L'ordine di derivazione n deve essere > 0");
    }

    // Numero di anelli necessari: N = ceil(n)
    int N = static_cast<int>(std::ceil(n));
    double n_sub = n / static_cast<double>(N);

    stages.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::abs(n_sub - 1.0) < 1e-6) {
            stages.push_back(MRR::first_order(R, xi, n_eff, n_g, df));
        } else {
            stages.push_back(
                MRR::fractional_order(n_sub, R, xi, n_eff, n_g, df));
        }
    }
}

MRRCascade MRRCascade::perturbed(const MRRCascade &nominal, double r, double xi,
                                 double n_eff, double n_g, double df) {
    MRRCascade casc;
    casc.total_order = nominal.total_order;
    casc.R = nominal.R;
    casc.xi = xi;
    casc.n_eff = n_eff;
    casc.n_g = n_g;
    casc.df = df;

    casc.stages.reserve(nominal.stages.size());
    for (size_t i = 0; i < nominal.stages.size(); ++i) {
        casc.stages.emplace_back(nominal.R, r, xi, n_eff, n_g, df);
    }
    return casc;
}

double MRRCascade::usable_band() const {
    if (stages.empty())
        return 0.0;
    double b0 = stages[0].usable_band();
    if (stages.size() == 1)
        return b0;

    // Con N anelli identici in serie, la larghezza a 3 dB scala come
    // sqrt(2^(1/N) - 1)
    double N = static_cast<double>(stages.size());
    return b0 * std::sqrt(std::pow(2.0, 1.0 / N) - 1.0);
}

std::string MRRCascade::description() const {
    char buf[128];
    if (stages.size() == 1) {
        std::snprintf(buf, sizeof(buf), "Single MRR (n = %.2f)", total_order);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "MRR Cascade (n = %.2f, %zu stages of n_i = %.2f)",
                      total_order, stages.size(), stage_order());
    }
    return std::string(buf);
}

std::string MRRCascade::label() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "Cascade (N=%zu, r=%.4f, xi=%.4f)",
                  stages.size(), stage_self_coupling(), xi);
    return std::string(buf);
}

std::string MRRCascade::params_string() const {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf),
        "Cascade: n = %.2f (%zu rings), R = %.0f um, r = %.4f, xi = %.4f\n"
        "tau = %.2f ps, usable band = %.2f GHz, df = %.3f GHz",
        total_order, stages.size(), R * 1e6, stage_self_coupling(), xi,
        round_trip_time() * 1e12, usable_band() / 1e9, df / 1e9);
    return std::string(buf);
}
