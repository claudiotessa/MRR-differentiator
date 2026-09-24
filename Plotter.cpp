#include "Plotter.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
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
    Eigen::Index i;
    ((freq_hz.abs() - std::abs(f_ref)).abs()).minCoeff(&i);
    const double ref = (mag(i) > 1e-300) ? mag(i) : mag.maxCoeff();
    return 20.0 * (mag / ref).log10();
}

void Plotter::show() { plt::show(); }

// =========================================================================
// TIME-DOMAIN FIGURES
// =========================================================================

void Plotter::plot_input_signal(const Simulation::Propagation &p) {
    plt::figure_size(800, 450);
    plt::plot(
        to_std_vec(p.time_ns), to_std_vec(p.in_norm),
        {{"color", "black"}, {"linewidth", "2"}, {"label", p.input_label}});

    if (p.view_ns > 0.0) {
        plt::xlim(-p.view_ns, p.view_ns);
    }
    finish_axes("Time [ns]", "Input signal y(t)");
}

void Plotter::plot_time_domain(const Simulation::Propagation &p) {
    plt::figure_size(950, 900);

    // Subplot 1: input signal
    plt::subplot(3, 1, 1);
    plt::plot(
        to_std_vec(p.time_ns), to_std_vec(p.in_norm),
        {{"color", "black"}, {"linewidth", "2"}, {"label", p.input_label}});
    if (p.view_ns > 0.0) {
        plt::xlim(-p.view_ns, p.view_ns);
    }
    finish_axes("Time [ns]", "Input signal y(t)");

    // Subplot 2: waveforms, ideal derivative vs ring output
    plt::subplot(3, 1, 2);
    plt::title(p.caption);
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.diff_real_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative"}});
    plt::plot(
        to_std_vec(p.time_ns), to_std_vec(p.ring_real_norm),
        {{"color", "red"}, {"linewidth", "2"}, {"label", "MRR output (real)"}});
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.ring_imag_norm),
              {{"color", "red"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "MRR output (imag)"}});
    if (p.view_ns > 0.0) {
        plt::xlim(-p.view_ns, p.view_ns);
    }
    finish_axes("Time [ns]", "Derivative y'(t)");

    // Subplot 3: optical power |y'(t)|^2
    plt::subplot(3, 1, 3);
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.power_diff),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative (power)"}});
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.power_ring),
              {{"color", "red"},
               {"linewidth", "2"},
               {"label", "MRR output (power)"}});
    if (p.view_ns > 0.0) {
        plt::xlim(-p.view_ns, p.view_ns);
    }
    finish_axes("Time [ns]", "|y'(t)|^2");

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

    // Algoritmo di Phase Unwrapping (elimina i salti di 2*pi)
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

    // Centra a zero a f = 0 per simmetria rispetto all'ideale
    Eigen::Index mid = casc_phase_raw.size() / 2;
    casc_phase_unwrapped -=
        (casc_phase_unwrapped(mid) + casc_phase_unwrapped(mid + 1)) / 2.0;

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
    plt::ylim(-30.0, 5.0);
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
    // plot_input_signal(p);
    plot_time_domain(p);
    plot_frequency_response(cascade);
    if (show_immediately)
        show();
}

void Plotter::plot_all(const Simulation &sim, bool show_immediately) {
    plot_all(sim.get_last_result(), sim.get_cascade(), show_immediately);
}

void Plotter::plot_monte_carlo(const MonteCarlo::Result &res,
                               double threshold) {
    plt::figure_size(850, 480);
    plt::hist(res.errors_Dn, 40, "steelblue", 0.75, true);

    plt::axvline(threshold, 0.0, 1.0,
                 {{"color", "red"},
                  {"linestyle", "--"},
                  {"linewidth", "2"},
                  {"label", "Soglia Yield (10%)"}});
    plt::axvline(res.mean_error, 0.0, 1.0,
                 {{"color", "orange"},
                  {"linestyle", "-"},
                  {"linewidth", "2"},
                  {"label", "Media Dn"}});

    char title[128];
    std::snprintf(title, sizeof(title),
                  "Distribuzione Errore Monte Carlo (Resa = %.1f%%)",
                  res.yield_rate);
    plt::title(std::string(title));
    finish_axes("Errore di derivazione D_n [%]", "Densita di probabilita");
    plt::tight_layout();
}
