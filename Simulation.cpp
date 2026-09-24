#include "Simulation.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <unsupported/Eigen/FFT>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;

// =========================================================================
// AUSILIARI GRAFICI INTERNI
// =========================================================================

// The ring is always the thick red line, the ideal the dashed blue one.
void Simulation::plot_ring_vs_ideal(const std::vector<double> &x,
                                    const std::vector<double> &ring,
                                    const std::vector<double> &ideal,
                                    const std::string &ring_label) const {
    plt::plot(x, ring,
              {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label}});
    plt::plot(x, ideal,
              {{"color", "blue"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "Ideal"}});
}

// Axis decoration every plot shares.
void Simulation::finish_axes(const std::string &xlabel,
                             const std::string &ylabel) const {
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

void Simulation::show() const { plt::show(); }

// =========================================================================
// SIGNAL HELPERS
// =========================================================================

// Allinea le curve per una migliore comparazione visiva
long Simulation::best_lag(const ArrayXd &a, const ArrayXd &b) {
    const long N = a.size();
    FFT<double> fft;
    VectorXcd A, B, corr;
    VectorXd av = (a - a.mean()).matrix();
    VectorXd bv = (b - b.mean()).matrix();
    fft.fwd(A, av);
    fft.fwd(B, bv);
    VectorXcd prod = B.array() * A.array().conjugate();
    fft.inv(corr, prod);

    Eigen::Index k;
    ArrayXd re = corr.real();
    re.maxCoeff(&k);
    return (k > N / 2) ? static_cast<long>(k) - N : static_cast<long>(k);
}

ArrayXd Simulation::shift_samples(const ArrayXd &arr, long k) {
    const long N = arr.size();
    ArrayXd out = ArrayXd::Zero(N);
    if (k >= 0)
        out.tail(N - k) = arr.head(N - k);
    else
        out.head(N + k) = arr.tail(N + k);
    return out;
}

double Simulation::power_error(const ArrayXd &out, const ArrayXd &ideal) {
    if (out.size() != ideal.size())
        throw std::invalid_argument(
            "power_error: the two waveforms must share a time axis");

    const double reference = ideal.sum();
    if (!(reference > 0.0))
        throw std::invalid_argument(
            "power_error: the ideal derivative carries no energy");

    return (out - ideal).abs().sum() / reference;
}

ArrayXd Simulation::to_dB(const ArrayXd &mag, const ArrayXd &freq_hz,
                          double f_ref) {
    Eigen::Index i;
    ((freq_hz.abs() - f_ref).abs()).minCoeff(&i);
    // A reference bin sitting on a null would send the whole curve to -inf.
    const double ref = (mag(i) > 1e-300) ? mag(i) : mag.maxCoeff();
    return 20.0 * (mag / ref).log10();
}

// =========================================================================
// INPUT PULSES
// =========================================================================

Simulation::Input Simulation::Input::gaussian(double T0) {
    Input in;
    in.shape = Gaussian;
    in.T0 = T0;
    return in;
}

Simulation::Input Simulation::Input::gaussian_matched(const MRR &ring,
                                                      double ratio) {
    // exp(-(t/T0)^2) transforms to exp(-(pi*f*T0)^2), whose amplitude FWHM is
    // 2*sqrt(ln2)/(pi*T0). Solve that for the wanted fraction of the band.
    const double band = ring.usable_band();
    if (!std::isfinite(band) || band <= 0.0)
        throw std::invalid_argument(
            "gaussian_matched: the ring has no usable band");
    if (!std::isfinite(ratio) || ratio <= 0.0)
        throw std::invalid_argument("gaussian_matched: ratio must be > 0");

    const double want = ratio * band;
    return gaussian(2.0 * std::sqrt(std::log(2.0)) / (M_PI * want));
}

Simulation::Input Simulation::Input::super_gaussian(double T0, int m) {
    Input in;
    in.shape = SuperGaussian;
    in.T0 = T0;
    in.m = m;
    return in;
}

Simulation::Input Simulation::Input::sech(double T0) {
    Input in;
    in.shape = Sech;
    in.T0 = T0;
    return in;
}

Simulation::Input Simulation::Input::rectangular(double T0) {
    Input in;
    in.shape = Rectangular;
    in.T0 = T0;
    return in;
}

ArrayXd Simulation::Input::sample(const ArrayXd &t) const {
    const ArrayXd u = t / T0;
    switch (shape) {
    case Gaussian:
        return (-u.square()).exp();
    case SuperGaussian:
        return (-u.abs().pow(2 * m)).exp();
    case Sech:
        return 1.0 / u.cosh();
    case Rectangular: {
        // A true step has infinite bandwidth and would alias on any grid, so
        // the edges are raised cosines one twentieth of the pulse wide.
        const double w = 0.05;
        ArrayXd e = ((u.abs() - 1.0) / w).min(1.0).max(-1.0);
        return 0.5 * (1.0 - (M_PI * e / 2.0).sin());
    }
    }
    return ArrayXd::Zero(t.size());
}

std::string Simulation::Input::describe() const {
    char buf[96];
    switch (shape) {
    case Gaussian:
        std::snprintf(buf, sizeof(buf), "Gaussian, T0 = %.3g ns", T0 * 1e9);
        break;
    case SuperGaussian:
        std::snprintf(buf, sizeof(buf),
                      "super-Gaussian (order %d), T0 = %.3g ns", 2 * m,
                      T0 * 1e9);
        break;
    case Sech:
        std::snprintf(buf, sizeof(buf), "sech, T0 = %.3g ns", T0 * 1e9);
        break;
    case Rectangular:
        std::snprintf(buf, sizeof(buf), "rectangular, T0 = %.3g ns", T0 * 1e9);
        break;
    }
    return std::string(buf);
}

// =========================================================================
// PROPAGAZIONE TEMPORALE (TIME DOMAIN COMPUTATION)
// =========================================================================

const Simulation::Propagation &Simulation::run(const Input &in, bool align) {
    // Without these an infinite T0 - which is what sizing a pulse against a
    // zero-width band used to produce - silently fills the time axis with NaN
    // and only aborts 100 lines later, inside std::pow, naming nothing.
    if (N < 2)
        throw std::invalid_argument("Simulation::run: N must be at least 2");
    if (!std::isfinite(in.T0) || in.T0 <= 0.0)
        throw std::invalid_argument(
            "Simulation::run: input T0 must be finite and > 0");

    const double ringdown =
        ring.round_trip_time() /
        (1.0 - ring.self_coupling() * ring.round_trip_loss());
    const double window = std::max(10.0 * in.T0, 40.0 * ringdown);
    if (!std::isfinite(window) || window <= 0.0)
        throw std::runtime_error(
            "Simulation::run: the time window is not finite");

    ArrayXd time = ArrayXd::LinSpaced(N, -window, window);
    const double dt = time(1) - time(0);

    const long half = N / 2;
    ArrayXcd Df = ((ArrayXd::LinSpaced(N, 0.0, static_cast<double>(N - 1)) -
                    static_cast<double>(half)) /
                   (static_cast<double>(N) * dt))
                      .cast<std::complex<double>>();

    ArrayXd E_in = in.sample(time);

    std::cout << "--- MRR configuration ---\n"
              << "order n:  " << n << '\n'
              << "r:        " << ring.self_coupling() << '\n'
              << "t:        " << ring.cross_coupling() << '\n'
              << "xi:       " << ring.round_trip_loss() << '\n'
              << "tau:      " << ring.round_trip_time() * 1e12 << " ps\n"
              << "band:     " << ring.usable_band() / 1e9 << " GHz\n"
              << "phase dv: " << ring.phase_transition_width() / 1e9 << " GHz\n"
              << "df:       " << ring.resonance_offset() / 1e9 << " GHz\n"
              << "input:    " << in.describe() << std::endl;

    ArrayXcd H_through = ring.compute_H(Df);

    ArrayXcd H_diff(N);
    for (long i = 0; i < N; ++i) {
        std::complex<double> j_omega(0.0, 2.0 * M_PI * Df(i).real());
        H_diff(i) = (std::abs(j_omega) < 1e-18) ? std::complex<double>(0.0, 0.0)
                                                : std::pow(j_omega, n);
    }

    Eigen::FFT<double> fft;
    VectorXcd fft_raw;
    VectorXd E_in_vec = E_in.matrix();
    fft.fwd(fft_raw, E_in_vec);

    VectorXcd in_spectrum = fftshift(fft_raw);

    VectorXcd out_ring_f = in_spectrum.array() * H_through;
    VectorXcd out_diff_f = in_spectrum.array() * H_diff;

    VectorXcd out_ring_t, out_diff_t;
    fft.inv(out_ring_t, fftshift(out_ring_f));
    fft.inv(out_diff_t, fftshift(out_diff_f));

    Propagation p;
    p.ring_label = ring.label();
    p.input_label = in.describe();
    p.view_ns = 2.5 * in.T0 * 1e9;

    p.power_ring = out_ring_t.array().abs2();
    p.power_diff = out_diff_t.array().abs2();

    p.time_ns = time * 1e9;
    p.in_norm = E_in / E_in.maxCoeff();

    p.diff_real_norm = out_diff_t.real().array();
    double max_diff = p.diff_real_norm.maxCoeff();
    if (std::abs(max_diff) > 1e-12)
        p.diff_real_norm /= max_diff;

    p.ring_real_norm = out_ring_t.real().array();
    p.ring_imag_norm = out_ring_t.imag().array();
    double max_ring = p.ring_real_norm.maxCoeff();
    if (std::abs(max_ring) > 1e-12) {
        p.ring_real_norm /= max_ring;
        p.ring_imag_norm /= max_ring;
    }

    if (p.power_diff.maxCoeff() > 1e-12)
        p.power_diff /= p.power_diff.maxCoeff();
    if (p.power_ring.maxCoeff() > 1e-12)
        p.power_ring /= p.power_ring.maxCoeff();

    p.lag = best_lag(p.power_diff, p.power_ring);
    p.lag_ps = p.lag * dt * 1e12;

    // Always measured aligned; `align` only decides what the figures show.
    const ArrayXd ideal_aligned = shift_samples(p.power_diff, p.lag);
    p.error_Dn = power_error(p.power_ring, ideal_aligned);

    char caption[224];
    std::snprintf(caption, sizeof(caption),
                  "ring lags the ideal by %+.1f ps (%+.1f tau), %s\n"
                  "D_n = %.2f %%",
                  p.lag_ps, p.lag * dt / ring.round_trip_time(),
                  align ? "ideal shifted onto the ring" : "shown unshifted",
                  p.error_Dn * 100.0);
    p.caption = std::string(caption);
    std::cout << p.caption << std::endl;

    if (align) {
        p.power_diff = ideal_aligned;
        p.diff_real_norm = shift_samples(p.diff_real_norm, p.lag);
    }

    last_propagation = p;
    has_result = true;
    return last_propagation;
}

void Simulation::plot_input_signal() const {
    if (!has_result) {
        throw std::runtime_error(
            "Nessun dato propagato. Esegui prima sim.run()!");
    }
    plt::figure_size(800, 450);
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.in_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", last_propagation.input_label}});
    if (last_propagation.view_ns > 0.0) {
        plt::xlim(-last_propagation.view_ns, last_propagation.view_ns);
    }
    finish_axes("Time [ns]", "Input signal y(t)");
}

// =========================================================================
// VISUALIZZAZIONE TEMPORALE UNIFICATA (3 SUBPLOT)
// =========================================================================

void Simulation::plot_time_domain() const {
    if (!has_result) {
        throw std::runtime_error(
            "Nessun dato propagato. Esegui prima sim.propagate()!");
    }

    plt::figure_size(950, 900);

    // Subplot 1: Segnale di ingresso
    plt::subplot(3, 1, 1);
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.in_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", last_propagation.input_label}});
    if (last_propagation.view_ns > 0.0) {
        plt::xlim(-last_propagation.view_ns, last_propagation.view_ns);
    }
    finish_axes("Time [ns]", "Input signal y(t)");

    // Subplot 2: Forme d'onda (derivata ideale vs uscita dell'anello)
    plt::subplot(3, 1, 2);
    plt::title(last_propagation.caption);
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.diff_real_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative"}});
    plt::plot(
        to_std_vec(last_propagation.time_ns),
        to_std_vec(last_propagation.ring_real_norm),
        {{"color", "red"}, {"linewidth", "2"}, {"label", "MRR output (real)"}});
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.ring_imag_norm),
              {{"color", "red"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "MRR output (imag)"}});
    if (last_propagation.view_ns > 0.0) {
        plt::xlim(-last_propagation.view_ns, last_propagation.view_ns);
    }
    finish_axes("Time [ns]", "Derivative y'(t)");

    // Subplot 3: Potenza ottica |y'(t)|^2
    plt::subplot(3, 1, 3);
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.power_diff),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative (power)"}});
    plt::plot(to_std_vec(last_propagation.time_ns),
              to_std_vec(last_propagation.power_ring),
              {{"color", "red"},
               {"linewidth", "2"},
               {"label", "MRR output (power)"}});
    if (last_propagation.view_ns > 0.0) {
        plt::xlim(-last_propagation.view_ns, last_propagation.view_ns);
    }
    finish_axes("Time [ns]", "|y'(t)|^2");

    plt::tight_layout();
}
// =========================================================================
// RISPOSTA IN FREQUENZA UNIFICATA
// =========================================================================

void Simulation::plot_frequency_response() const {
    // 1. La banda di differenziazione viene letta direttamente dall'anello.
    // Eq. (4) is not usable here: it collapses to 0 at critical coupling, which
    // would put the dB anchor exactly on the notch null and send `span_hz` to
    // its floor.
    const double band = ring.usable_band();
    if (!std::isfinite(band) || band <= 0.0)
        throw std::runtime_error(
            "plot_frequency_response: the ring has no usable band");

    // Anchor well inside the band but clear of the null at the resonance.
    const double f_ref = band / 4.0;

    // 2. Griglia di frequenza centrata sul notch
    const double span_hz = 3.0 * band;
    ArrayXd freq_hz = ArrayXd::LinSpaced(N, -span_hz, span_hz);
    std::vector<double> freq_ghz = to_std_vec((freq_hz / 1e9).eval());

    // --- MRR ---
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

    // Linee di ancoraggio su +/- band/4
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

void Simulation::plot() const {
    plot_input_signal();
    plot_time_domain();
    plot_frequency_response();
    show();
}
