#include "MRRCascade.hpp"

#include <vector>

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

MRRCascade::MRRCascade(double n, double R, double xi, double n_eff, double n_g,
                       double df)
    : total_order(n), R(R), xi(xi), n_eff(n_eff), n_g(n_g), df(df) {
    if (n <= 0.0) {
        throw std::invalid_argument("MRRCascade: the order n must be > 0");
    }

    // Rings needed: N = ceil(n), each carrying an equal share of the order.
    // The tolerance keeps a computed 1.0000000000000002 at one ring.
    int N = static_cast<int>(std::ceil(n - 1e-9));
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

MRRCascade
MRRCascade::perturbed(const MRRCascade &nominal,
                      const std::vector<StageParams> &stage_params) {
    if (stage_params.size() != nominal.stages.size())
        throw std::invalid_argument(
            "MRRCascade::perturbed: one parameter set per stage");

    MRRCascade casc;
    casc.total_order = nominal.total_order;
    casc.stages.reserve(stage_params.size());
    for (const StageParams &p : stage_params) {
        casc.stages.emplace_back(p.R, p.r, p.xi, p.n_eff, p.n_g, p.df);
        casc.R += p.R;
        casc.xi += p.xi;
        casc.n_eff += p.n_eff;
        casc.n_g += p.n_g;
        casc.df += p.df;
    }
    const double N = static_cast<double>(stage_params.size());
    casc.R /= N;
    casc.xi /= N;
    casc.n_eff /= N;
    casc.n_g /= N;
    casc.df /= N;
    return casc;
}

MRRCascade MRRCascade::perturbed(const MRRCascade &nominal, double r, double xi,
                                 double n_eff, double n_g, double df,
                                 double R_custom) {
    const double R = (R_custom > 0.0) ? R_custom : nominal.R;
    return perturbed(nominal, std::vector<StageParams>(nominal.stages.size(),
                                                       {R, r, xi, n_eff, n_g,
                                                        df}));
}

double MRRCascade::usable_band(double tol_dB) const {
    if (stages.empty())
        return 0.0;

    const double fsr = 1.0 / stages[0].round_trip_time();
    // Well inside the single ring's resonance, where the power law certainly
    // holds; this fixes the constant C of the comparison.
    const double f_ref = stages[0].usable_band() / 4.0;

    const int M = 4000;
    const double f_lo = fsr * 1e-5, f_hi = fsr * 0.5;
    const double step = std::pow(f_hi / f_lo, 1.0 / (M - 1));

    Eigen::ArrayXd f(M);
    for (int i = 0; i < M; ++i)
        f(i) = f_lo * std::pow(step, i);
    const Eigen::ArrayXd mag = compute_H(f).abs();

    // Ideal power law, normalised at f_ref.
    int k_ref = 0;
    for (int i = 1; i < M; ++i)
        if (std::abs(f(i) - f_ref) < std::abs(f(k_ref) - f_ref))
            k_ref = i;
    const double C = mag(k_ref) / std::pow(2.0 * M_PI * f(k_ref), total_order);

    auto within = [&](int i) {
        const double ideal = C * std::pow(2.0 * M_PI * f(i), total_order);
        if (!(ideal > 0.0) || !(mag(i) > 0.0))
            return false;
        return std::abs(20.0 * std::log10(mag(i) / ideal)) <= tol_dB;
    };

    if (!within(k_ref))
        return stages[0].usable_band(); // nothing to measure, fall back

    int lo = k_ref, hi = k_ref;
    while (lo > 0 && within(lo - 1))
        --lo;
    while (hi < M - 1 && within(hi + 1))
        ++hi;
    return f(hi) - f(lo);
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
        "Cascade: n = %.2f (%zu ring%s), R = %.2f um, r = %.4f, xi = %.4f\n"
        "tau = %.2f ps, usable band = %.1f GHz, df = %.3f GHz",
        total_order, stages.size(), stages.size() == 1 ? "" : "s", R * 1e6, stage_self_coupling(), xi,
        round_trip_time() * 1e12, usable_band() / 1e9, df / 1e9);
    return std::string(buf);
}
