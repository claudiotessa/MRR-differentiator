#include "Plotter.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;

void Plotter::plot_ring_vs_ideal(const std::vector<double> &x,
                                 const std::vector<double> &ring_data,
                                 const std::vector<double> &ideal_data,
                                 const std::string &ring_label) {
    plt::plot(x, ring_data,
              {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label}});
    plt::plot(x, ideal_data,
              {{"color", "blue"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "Ideal"}});
}

void Plotter::finish_axes(const std::string &xlabel,
                          const std::string &ylabel) {
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

Eigen::ArrayXd Plotter::to_dB(const Eigen::ArrayXd &mag,
                              const Eigen::ArrayXd &freq_hz, double f_ref) {
    // |2*pi*f|^n has units of s^-n, so ring and ideal only compare once both
    // are pinned to 0 dB at the same frequency. Anchor on +f_ref explicitly:
    // picking it by |f| lands on whichever side the grid happens to hit
    // first, and a detuned ring is not symmetric.
    const double target = std::abs(f_ref);
    Eigen::Index i = -1;
    double best = std::numeric_limits<double>::infinity();
    for (Eigen::Index k = 0; k < freq_hz.size(); ++k) {
        if (freq_hz(k) <= 0.0)
            continue;
        const double d = std::abs(freq_hz(k) - target);
        if (d < best) {
            best = d;
            i = k;
        }
    }
    if (i < 0 || !(mag(i) > 0.0))
        throw std::runtime_error(
            "Plotter::to_dB: no usable reference bin at +f_ref - renormalising "
            "the two curves differently would make them incomparable");

    // The ideal is exactly 0 at DC; floor it so log10 stays finite. 1e-12 of
    // the reference is 240 dB down, far below any axis we draw.
    const double floor_mag = mag(i) * 1e-12;
    return 20.0 * (mag.max(floor_mag) / mag(i)).log10();
}

void Plotter::show() { plt::show(); }

void Plotter::headless() { plt::backend("Agg"); }

// =========================================================================
// FIGURE FILES
// =========================================================================

namespace {
std::string g_out_dir = "fig";
}

void Plotter::set_output_dir(const std::string &dir) { g_out_dir = dir; }

void Plotter::save(const std::string &stem) {
    std::filesystem::create_directories(g_out_dir);
    const std::string path =
        (std::filesystem::path(g_out_dir) / (stem + ".pdf")).string();
    plt::save(path);
    plt::close();
    std::printf("wrote %s\n", path.c_str());
}

// =========================================================================
// TIME-DOMAIN FIGURES
// =========================================================================

void Plotter::panel_input(const Simulation::Propagation &p) {
    plt::plot(
        to_std_vec(p.time_ps), to_std_vec(p.in_norm),
        {{"color", "black"}, {"linewidth", "2"}, {"label", p.input_label}});
    if (p.view_ps > 0.0) {
        plt::xlim(-p.view_ps, p.view_ps);
    }
    finish_axes("Time [ps]", "Input y(t) [norm.]");
}

void Plotter::panel_field(const Simulation::Propagation &p) {
    plt::title(p.caption);
    plt::plot(to_std_vec(p.time_ps), to_std_vec(p.diff_real_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative"}});
    plt::plot(
        to_std_vec(p.time_ps), to_std_vec(p.ring_real_norm),
        {{"color", "red"}, {"linewidth", "2"}, {"label", "MRR output (real)"}});
    plt::plot(to_std_vec(p.time_ps), to_std_vec(p.ring_imag_norm),
              {{"color", "red"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "MRR output (imag)"}});
    if (p.view_ps > 0.0) {
        plt::xlim(-p.view_ps, p.view_ps);
    }
    finish_axes("Time [ps]", "Derivative y'(t) [norm.]");
}

// Optical power, which is what D_n is measured on.
void Plotter::panel_power(const Simulation::Propagation &p) {
    plt::plot(to_std_vec(p.time_ps), to_std_vec(p.power_diff),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative (power)"}});
    plt::plot(to_std_vec(p.time_ps), to_std_vec(p.power_ring),
              {{"color", "red"},
               {"linewidth", "2"},
               {"label", "MRR output (power)"}});
    if (p.view_ps > 0.0) {
        plt::xlim(-p.view_ps, p.view_ps);
    }
    finish_axes("Time [ps]", "Power |y'(t)|^2 [norm.]");
}

void Plotter::plot_time_domain(const Simulation::Propagation &p) {
    plt::figure_size(950, 900);
    plt::subplot(3, 1, 1);
    panel_input(p);
    plt::subplot(3, 1, 2);
    panel_field(p);
    plt::subplot(3, 1, 3);
    panel_power(p);
    plt::tight_layout();
}

void Plotter::plot_time_power(const Simulation::Propagation &p) {
    plt::figure_size(950, 650);
    plt::subplot(2, 1, 1);
    panel_input(p);
    plt::subplot(2, 1, 2);
    plt::title(p.caption);
    panel_power(p);
    plt::tight_layout();
}

// =========================================================================
// FREQUENCY-DOMAIN FIGURES
// =========================================================================

void Plotter::plot_frequency_response(const MRRCascade &cascade, long N) {
    const double band = cascade.usable_band();
    const double n = cascade.order();
    if (!std::isfinite(band) || band <= 0.0) {
        throw std::runtime_error(
            "Plotter::frequency_response: cascade has no usable band");
    }

    const double f_ref = band / 4.0;
    const double span_hz = 3.0 * band;
    ArrayXd freq_hz = ArrayXd::LinSpaced(N, -span_hz, span_hz);
    std::vector<double> freq_ghz = to_std_vec((freq_hz / 1e9).eval());

    ArrayXcd H_casc = cascade.compute_H(freq_hz);
    ArrayXd casc_dB = to_dB(H_casc.abs(), freq_hz, f_ref);
    ArrayXd casc_phase_raw = cascade.compute_phase(freq_hz);

    // Phase unwrapping: remove the 2*pi jumps.
    ArrayXd casc_phase_unwrapped = casc_phase_raw;
    double offset = 0.0;
    for (long i = 1; i < casc_phase_raw.size(); ++i) {
        double diff = casc_phase_raw(i) - casc_phase_raw(i - 1);
        if (diff > M_PI) {
            offset -= 2.0 * M_PI;
        } else if (diff < -M_PI) {
            offset += 2.0 * M_PI;
        }
        casc_phase_unwrapped(i) += offset;
    }

    // Centre on f = 0. The grid does not land exactly on zero, so mid-1 and
    // mid are the two samples straddling it.
    Eigen::Index mid = casc_phase_raw.size() / 2;
    casc_phase_unwrapped -=
        (casc_phase_unwrapped(mid - 1) + casc_phase_unwrapped(mid)) / 2.0;

    ArrayXd casc_phase = casc_phase_unwrapped / M_PI;
    ArrayXd ideal_abs = (2.0 * M_PI * freq_hz).abs().pow(n);
    ArrayXd ideal_dB = to_dB(ideal_abs, freq_hz, f_ref);
    ArrayXd ideal_phase = freq_hz.sign() * (n / 2.0);

    char heading[96];
    std::snprintf(heading, sizeof(heading),
                  "Differentiator response (order n = %.2f)", n);

    plt::figure_size(1100, 480);

    plt::subplot(1, 2, 1);
    plt::title(std::string(heading) + "\n" + cascade.params_string());
    plot_ring_vs_ideal(freq_ghz, to_std_vec(casc_dB), to_std_vec(ideal_dB),
                       cascade.label());
    plt::axvline(f_ref / 1e9, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", ":"}});
    plt::axvline(-f_ref / 1e9, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", ":"}});
    plt::xlim(-span_hz / 1e9, span_hz / 1e9);
    // The ideal climbs as |f|^n without bound, so the top of the axis has to
    // follow the span; a fixed +5 dB clipped it even on the narrow view.
    plt::ylim(-40.0, std::max(5.0, ideal_dB.maxCoeff() + 5.0));
    finish_axes("Frequency [GHz]", "Magnitude [dB]");

    plt::subplot(1, 2, 2);
    char title[96];
    std::snprintf(title, sizeof(title), "Phase (ideal step = +/- %.2f pi)",
                  n / 2.0);
    plt::title(std::string(title));
    plot_ring_vs_ideal(freq_ghz, to_std_vec(casc_phase),
                       to_std_vec(ideal_phase), cascade.label());
    plt::xlim(-span_hz / 1e9, span_hz / 1e9);
    plt::ylim(-1.0 * std::ceil(n), 1.0 * std::ceil(n));
    finish_axes("Frequency [GHz]", "Phase [rad / pi]");

    plt::tight_layout();
}

void Plotter::plot_all(const Simulation::Propagation &p,
                       const MRRCascade &cascade, bool show_immediately) {
    plot_time_domain(p);
    plot_frequency_response(cascade);
    if (show_immediately) {
        show();
    }
}

void Plotter::plot_all(const Simulation &sim, bool show_immediately) {
    plot_all(sim.get_last_result(), sim.get_cascade(), show_immediately);
}

void Plotter::plot_monte_carlo(const MonteCarlo::Result &res,
                               double threshold) {
    plt::figure_size(850, 480);
    // The fifth argument of hist() is `cumulative`, not `density`.
    plt::hist(res.errors_Dn, 40, "steelblue", 0.75, false);

    char thr_label[64];
    std::snprintf(thr_label, sizeof(thr_label), "yield threshold (%.0f%%)",
                  threshold);
    plt::axvline(threshold, 0.0, 1.0,
                 {{"color", "red"},
                  {"linestyle", "--"},
                  {"linewidth", "2"},
                  {"label", thr_label}});

    char mean_label[64];
    std::snprintf(mean_label, sizeof(mean_label), "mean D_n (%.2f%%)",
                  res.mean_error);
    plt::axvline(res.mean_error, 0.0, 1.0,
                 {{"color", "orange"},
                  {"linestyle", "-"},
                  {"linewidth", "2"},
                  {"label", mean_label}});

    char title[160];
    std::snprintf(title, sizeof(title),
                  "Monte Carlo error distribution, %zu devices"
                  " (yield = %.1f%%)",
                  res.errors_Dn.size(), res.yield_rate);
    plt::title(std::string(title));
    finish_axes("Differentiation error D_n [%]", "Devices");
    plt::tight_layout();
}

// =========================================================================
// PRESENTATION FIGURES
// =========================================================================

void Plotter::draw_iso_orders(const std::vector<double> &orders, double n_bold,
                              double xi_lo, double xi_hi) {
    // semilogy takes no keywords, so it only switches the axes to log; the
    // curves go through plot(), which keeps the scale.
    plt::semilogy(std::vector<double>{xi_lo}, std::vector<double>{1e-3}, "");

    const int M = 200;
    for (double n : orders) {
        std::vector<double> xs, ys;
        for (int k = 0; k < M; ++k) {
            const double x = xi_lo + (xi_hi - xi_lo) * k / (M - 1);
            // R, n_eff and n_g do not enter r, only Eq. (2) does.
            const double r =
                MRR::fractional_order(n, 1e-6, x, 1.0, 1.0).self_coupling();
            xs.push_back(x);
            ys.push_back(r - x);
        }
        const bool bold = std::abs(n - n_bold) < 1e-9;
        char label[32];
        std::snprintf(label, sizeof(label), "n = %.2g", n);
        if (bold) {
            plt::plot(xs, ys,
                      {{"color", "blue"}, {"linewidth", "3"}, {"label", label}});
        } else {
            plt::plot(xs, ys, {{"color", "gray"}, {"linewidth", "1"}});
            plt::text(xs.back(), ys.back(), std::string(" ") + label);
        }
    }
}

void Plotter::plot_locus_map(const std::vector<double> &orders, double n_bold,
                             double r_design, double xi_design) {
    plt::figure_size(900, 600);
    draw_iso_orders(orders, n_bold, 0.80, 0.995);
    plt::plot(std::vector<double>{xi_design},
              std::vector<double>{r_design - xi_design},
              {{"color", "red"},
               {"marker", "*"},
               {"markersize", "16"},
               {"linestyle", "none"},
               {"label", "design point [LIU25]"}});
    plt::xlim(0.80, 1.0);
    plt::title("Same-order curves of Eq. (2), under-coupled branch r > xi");
    finish_axes("Single-pass transmission xi", "r - xi");
    plt::tight_layout();
}

void Plotter::plot_dn_along_locus(const std::vector<double> &r,
                                  const std::vector<double> &D_fixed,
                                  const std::vector<double> &D_scaled, double n,
                                  double r_design) {
    plt::figure_size(900, 520);
    plt::plot(r, D_fixed,
              {{"color", "red"},
               {"linewidth", "2.5"},
               {"marker", "o"},
               {"markersize", "4"},
               {"label", "fixed 3 ps pulse"}});
    plt::plot(r, D_scaled,
              {{"color", "blue"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "pulse scaled with the FWHM"}});
    plt::axvline(r_design, 0.0, 1.0,
                 {{"color", "gray"},
                  {"linestyle", ":"},
                  {"linewidth", "2"},
                  {"label", "design point [LIU25]"}});
    char title[96];
    std::snprintf(title, sizeof(title), "D_n along the n = %.2f design curve", n);
    plt::title(title);
    finish_axes("Self-coupling r", "D_n [%]");
    plt::tight_layout();
}

void Plotter::plot_dn_vs_order(const std::vector<double> &n,
                               const std::vector<double> &D, double T0_ps,
                               const std::vector<double> &bench_n,
                               const std::vector<double> &bench_ours,
                               const std::vector<double> &bench_paper,
                               double threshold) {
    plt::figure_size(900, 520);
    char label[64];
    std::snprintf(label, sizeof(label), "ours, %.0f ps pulse", T0_ps);
    plt::plot(n, D, {{"color", "red"}, {"linewidth", "2.5"}, {"label", label}});
    plt::plot(bench_n, bench_ours,
              {{"color", "red"},
               {"marker", "o"},
               {"markersize", "9"},
               {"linestyle", "none"},
               {"label", "ours, paper's pulse"}});
    std::snprintf(label, sizeof(label), "%.0f%% bar", threshold);
    plt::axhline(threshold, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", "--"}, {"label", label}});
    // A ring is added at every integer order.
    for (double k : {1.0, 2.0})
        plt::axvline(k, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
    plt::title("D_n against the order, uniform cascade of ceil(n) rings");
    finish_axes("Order n", "D_n [%]");
    plt::tight_layout();
}

void Plotter::plot_mc_scatter(const MonteCarlo::Result &res, double n,
                              double r_design, double xi_design,
                              double threshold) {
    // The plane is one ring's: a cascade is placed by its per-ring order,
    // and each device by its ring-averaged r and xi.
    const double n_ring = n / std::ceil(n - 1e-9);

    std::vector<double> xp, yp, xf, yf;
    long off_axis = 0; // mean r <= mean xi has no place on a log axis
    for (size_t i = 0; i < res.errors_Dn.size(); ++i) {
        const double d = res.r_samples[i] - res.xi_samples[i];
        if (!(d > 0.0)) {
            ++off_axis;
            continue;
        }
        const bool pass = res.errors_Dn[i] <= threshold;
        (pass ? xp : xf).push_back(res.xi_samples[i]);
        (pass ? yp : yf).push_back(d);
    }

    // Frame the cloud, not the whole map.
    double lo = xi_design, hi = xi_design;
    for (double x : res.xi_samples) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    const double pad = 0.2 * (hi - lo);
    lo -= pad;
    hi += pad;

    plt::figure_size(900, 600);
    std::vector<double> orders;
    for (int k = -2; k <= 2; ++k)
        if (n_ring + 0.1 * k > 0.0 && n_ring + 0.1 * k < 1.0)
            orders.push_back(n_ring + 0.1 * k);
    draw_iso_orders(orders, n_ring, lo, hi);

    char label[64];
    std::snprintf(label, sizeof(label), "D_n <= %.0f%% (%zu)", threshold,
                  xp.size());
    plt::scatter(xp, yp, 12.0, {{"color", "green"}, {"label", label}});
    std::snprintf(label, sizeof(label), "D_n > %.0f%% (%zu)", threshold,
                  xf.size());
    plt::scatter(xf, yf, 12.0, {{"color", "red"}, {"label", label}});
    plt::plot(std::vector<double>{xi_design},
              std::vector<double>{r_design - xi_design},
              {{"color", "black"},
               {"marker", "*"},
               {"markersize", "16"},
               {"linestyle", "none"},
               {"label", "design point"}});
    plt::xlim(lo, hi);

    // Over-coupled counts devices with any ring at r <= xi; off_axis only
    // those whose average is, so the two can differ for a cascade.
    char title[160];
    std::snprintf(title, sizeof(title),
                  "Monte Carlo devices, n = %.2f (%.2f per ring), yield %.1f%%\n"
                  "%ld over-coupled, %ld off-axis (mean r <= mean xi)",
                  n, n_ring, res.yield_rate, res.over_coupled, off_axis);
    plt::title(title);
    finish_axes("Single-pass transmission xi", "r - xi");
    plt::tight_layout();
}

void Plotter::plot_yield_along_locus(const std::vector<double> &r,
                                     const std::vector<double> &D_nominal,
                                     const std::vector<double> &yield,
                                     double n, double r_design,
                                     double threshold) {
    plt::figure_size(900, 650);

    plt::subplot(2, 1, 1);
    char title[96];
    std::snprintf(title, sizeof(title), "Along the n = %.2f design curve", n);
    plt::title(title);
    plt::plot(r, D_nominal,
              {{"color", "red"},
               {"marker", "o"},
               {"linewidth", "2"},
               {"label", "nominal D_n"}});
    char label[32];
    std::snprintf(label, sizeof(label), "%.0f%% bar", threshold);
    plt::axhline(threshold, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", "--"}, {"label", label}});
    plt::axvline(r_design, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
    finish_axes("Self-coupling r", "D_n [%]");

    plt::subplot(2, 1, 2);
    plt::plot(r, yield,
              {{"color", "blue"},
               {"marker", "o"},
               {"linewidth", "2"},
               {"label", "Monte Carlo yield"}});
    plt::axvline(r_design, 0.0, 1.0,
                 {{"color", "gray"},
                  {"linestyle", ":"},
                  {"label", "design point [LIU25]"}});
    plt::ylim(0.0, 100.0);
    finish_axes("Self-coupling r", "Yield [%]");
    plt::tight_layout();
}

void Plotter::plot_tolerance_breakdown(const std::vector<std::string> &labels,
                                       const std::vector<double> &yield,
                                       double threshold) {
    std::vector<double> x(yield.size());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<double>(i);

    plt::figure_size(900, 480);
    plt::bar(x, yield, "black", "-", 1.0, {{"color", "steelblue"}});
    plt::xticks(x, labels);
    for (size_t i = 0; i < x.size(); ++i) {
        char v[16];
        std::snprintf(v, sizeof(v), "%.1f%%", yield[i]);
        plt::text(x[i] - 0.2, yield[i] + 1.5, v);
    }
    plt::ylim(0.0, 110.0);
    char title[96];
    std::snprintf(title, sizeof(title),
                  "Yield (D_n <= %.0f%%) with one tolerance at a time",
                  threshold);
    plt::title(title);
    plt::ylabel("Yield [%]");
    plt::grid(true);
    plt::tight_layout();
}
