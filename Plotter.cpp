#include "Plotter.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;

// =========================================================================
// METODI PRIVATI AUSILIARI
// =========================================================================

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
// GRAFICI TEMPORALI
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

    // Subplot 1: Segnale di ingresso
    plt::subplot(3, 1, 1);
    plt::plot(
        to_std_vec(p.time_ns), to_std_vec(p.in_norm),
        {{"color", "black"}, {"linewidth", "2"}, {"label", p.input_label}});
    if (p.view_ns > 0.0) {
        plt::xlim(-p.view_ns, p.view_ns);
    }
    finish_axes("Time [ns]", "Input signal y(t)");

    // Subplot 2: Forme d'onda (derivata ideale vs uscita dell'anello)
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

    // Subplot 3: Potenza ottica |y'(t)|^2
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
// GRAFICI IN FREQUENZA
// =========================================================================

void Plotter::plot_frequency_response(const MRR &ring, double n, long N) {
    const double band = ring.usable_band();
    if (!std::isfinite(band) || band <= 0.0) {
        throw std::runtime_error(
            "Plotter::frequency_response: the ring has no usable band");
    }

    const double f_ref = band / 4.0;
    const double span_hz = 3.0 * band;
    ArrayXd freq_hz = ArrayXd::LinSpaced(N, -span_hz, span_hz);
    std::vector<double> freq_ghz = to_std_vec((freq_hz / 1e9).eval());

    // --- Calcolo risposta MRR ---
    ArrayXcd H_ring = ring.compute_H(freq_hz);
    ArrayXd ring_dB = to_dB(H_ring.abs(), freq_hz, f_ref);
    ArrayXd ring_phase = ring.compute_phase(freq_hz) / M_PI;

    // --- Derivata ideale (j*2*pi*f)^n ---
    ArrayXd ideal_abs = (2.0 * M_PI * freq_hz).abs().pow(n);
    ArrayXd ideal_dB = to_dB(ideal_abs, freq_hz, f_ref);
    ArrayXd ideal_phase = freq_hz.sign() * (n / 2.0);

    char title[192];
    char heading[96];
    if (std::abs(n - 1.0) < 1e-4) {
        std::snprintf(heading, sizeof(heading), "First order (n = 1.0)");
    } else {
        std::snprintf(heading, sizeof(heading), "Fractional order (n = %.2f)",
                      n);
    }

    plt::figure_size(1100, 480);

    // --- Subplot Modulo ---
    plt::subplot(1, 2, 1);
    plt::title(std::string(heading) + "\n" + ring.params_string());
    plot_ring_vs_ideal(freq_ghz, to_std_vec(ring_dB), to_std_vec(ideal_dB),
                       ring.label());

    plt::axvline(f_ref / 1e9, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", ":"}});
    plt::axvline(-f_ref / 1e9, 0.0, 1.0,
                 {{"color", "gray"}, {"linestyle", ":"}});

    const double view_ghz = span_hz / 1e9;
    plt::xlim(-view_ghz, view_ghz);
    plt::ylim(-25.0, 5.0);
    finish_axes("Frequency [GHz]", "Magnitude [dB]");

    // --- Subplot Fase ---
    std::snprintf(title, sizeof(title), "Phase (ideal = +/- %.2f pi)", n / 2.0);
    plt::subplot(1, 2, 2);
    plt::title(std::string(title));
    plot_ring_vs_ideal(freq_ghz, to_std_vec(ring_phase),
                       to_std_vec(ideal_phase), ring.label());

    plt::xlim(-view_ghz, view_ghz);
    plt::ylim(-1.0, 1.0);
    finish_axes("Frequency [GHz]", "Phase [rad / pi]");

    plt::tight_layout();
}

void Plotter::plot_all(const Simulation::Propagation &p, const MRR &ring,
                       double n, bool show_immediately) {
    plot_input_signal(p);
    plot_time_domain(p);
    plot_frequency_response(ring, n);

    if (show_immediately) {
        show();
    }
}

void Plotter::plot_all(const Simulation &sim, bool show_immediately) {
    plot_all(sim.get_last_result(), sim.get_ring(), sim.get_order(),
             show_immediately);
}
