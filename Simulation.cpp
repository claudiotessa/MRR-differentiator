#include "Simulation.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <unsupported/Eigen/FFT>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;

namespace {

// The ring is always the thick red line, the ideal the dashed blue one.
void plot_ring_vs_ideal(const std::vector<double> &x,
                        const std::vector<double> &ring,
                        const std::vector<double> &ideal,
                        const std::string &ring_label) {
    plt::plot(x, ring,
        {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label}});
    plt::plot(x, ideal,
        {{"color", "blue"},
        {"linestyle", "--"},
        {"linewidth", "2"},
        {"label", "Ideal"}});
}

// Axis decoration every plot shares.
void finish_axes(const std::string &xlabel, const std::string &ylabel) {
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

} // namespace

// =========================================================================
// SIGNAL HELPERS
// =========================================================================

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

ArrayXd Simulation::to_dB(const ArrayXd &mag, const ArrayXd &freq_hz,
                          double f_ref) {
    Eigen::Index i;
    ((freq_hz.abs() - f_ref).abs()).minCoeff(&i);
    return 20.0 * (mag / mag(i)).log10();
}

void Simulation::show() { plt::show(); }

// =========================================================================
// FREQUENCY DOMAIN
// =========================================================================

void Simulation::response(const MRR &m, double n, double B,
                          const std::string &heading) {
    // Frequency grid from -20 GHz to +20 GHz
    const long N = 10000;
    ArrayXcd Df =
        ArrayXd::LinSpaced(N, -20e9, 20e9).cast<std::complex<double>>();
    ArrayXd freq_hz = Df.real();
    std::vector<double> freq_ghz = to_std_vec((freq_hz / 1e9).eval());

    // The curves are normalised at B/2, the edge of the band the ring is
    // designed for.
    const double f_ref = B / 2.0;

    // --- Ring ---
    ArrayXcd H_ring = m.compute_H(Df);
    ArrayXd ring_dB = to_dB(H_ring.abs(), freq_hz, f_ref);
    ArrayXd ring_phase = m.compute_phase(Df) / M_PI;

    // --- Ideal n-th order derivative, (j*2*pi*f)^n ---
    ArrayXd ideal_abs = (2.0 * M_PI * freq_hz).abs().pow(n);
    ArrayXd ideal_dB = to_dB(ideal_abs, freq_hz, f_ref);
    // Phase is +n*pi/2 for f > 0 and -n*pi/2 for f < 0
    ArrayXd ideal_phase = freq_hz.sign() * (n / 2.0);

    char title[192];

    plt::figure_size(1100, 480);

    // --- Magnitude ---
    plt::subplot(1, 2, 1);
    plt::title(heading + "\n" + m.params_string());
    plot_ring_vs_ideal(freq_ghz, to_std_vec(ring_dB), to_std_vec(ideal_dB),
        m.label());
    plt::axvline(f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
    plt::axvline(-f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
    plt::ylim(-25.0, 15.0);
    finish_axes("Frequency [GHz]", "Magnitude [dB]");

    // --- Phase ---
    std::snprintf(title, sizeof(title), "Phase (ideal = +/- %.2f pi)", n / 2.0);
    plt::subplot(1, 2, 2);
    plt::title(std::string(title));
    plot_ring_vs_ideal(freq_ghz, to_std_vec(ring_phase), to_std_vec(ideal_phase),
        m.label());
    plt::ylim(-1.0, 1.0);
    finish_axes("Frequency [GHz]", "Phase [rad / pi]");

    plt::tight_layout();
}

void Simulation::first_order_response(const MRR &m, double B) {
    char heading[96];
    std::snprintf(heading, sizeof(heading),
        "First order (n = 1, target B = %.1f GHz)", B / 1e9);
    response(m, 1.0, B, std::string(heading));
}

void Simulation::fractional_response(const MRR &m, double n, double B) {
    char heading[96];
    std::snprintf(heading, sizeof(heading), "Fractional order (n = %.2f)", n);
    response(m, n, B, std::string(heading));
}

// =========================================================================
// TIME DOMAIN
// =========================================================================

Simulation::Propagation Simulation::propagate(const MRR &ring, double n,
                                              bool align) {
    // --- Time axis and matching FFT frequency grid ---
    const long N = 100000;
    ArrayXd time = ArrayXd::LinSpaced(N, -10e-9, 10e-9); // -10 ns to +10 ns
    const double dt = time(1) - time(0);

    // Frequency axis centred on 0. The bin spacing must be exactly 1/(N*dt) to
    // line up with the FFT output; LinSpaced over [-1/(2dt), +1/(2dt)] would
    // stretch the grid by N/(N-1) and leave the last bin half a bin off.
    const long half = N / 2; // N is even, so bin `half` is the most negative one
    ArrayXcd Df =
        ((ArrayXd::LinSpaced(N, 0.0, static_cast<double>(N - 1)) -
          static_cast<double>(half)) /
            (static_cast<double>(N) * dt))
            .cast<std::complex<double>>();

    // --- Input signal: 12th-order super-Gaussian, T0 = 1 ns ---
    const double A = 1e10;
    ArrayXd E_in = (-((0.1 * A * time).pow(12))).exp();

    std::cout << "--- Fractional MRR configuration ---\n"
              << "order n: " << n << '\n'
              << "r:       " << ring.self_coupling() << '\n'
              << "t:       " << ring.cross_coupling() << '\n'
              << "xi:      " << ring.round_trip_loss() << '\n'
              << "tau:     " << ring.round_trip_time() * 1e12 << " ps"
              << std::endl;

    // --- Frequency responses: physical ring vs ideal fractional derivative ---
    ArrayXcd H_through = ring.compute_H(Df);

    // Ideal fractional derivative (j*2*pi*Df)^n. No delay term here: the two
    // waveforms are aligned afterwards with best_lag(), see the note there.
    ArrayXcd H_diff(N);
    for (long i = 0; i < N; ++i) {
        std::complex<double> j_omega(0.0, 2.0 * M_PI * Df(i).real());
        H_diff(i) = (std::abs(j_omega) < 1e-18)
                        ? std::complex<double>(0.0, 0.0)
                        : std::pow(j_omega, n);
    }

    // --- Propagation: FFT, spectral multiplication, IFFT ---
    FFT<double> fft;
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

    // Optical power |y(t)|^2
    p.power_ring = out_ring_t.array().abs2();
    p.power_diff = out_diff_t.array().abs2();

    // --- Normalisation for plotting ---
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

    // The ideal differentiator has zero group delay, so the offset between the
    // two waveforms is a real property of the ring. The lag is always measured
    // and reported; `align` only decides whether it is also removed. Sliding
    // the ideal onto the ring answers a different question - "is the derivative
    // the right shape?" rather than "when does it arrive?" - by taking the
    // latency out so only shape error remains.
    p.lag = best_lag(p.power_diff, p.power_ring);
    p.lag_ps = p.lag * dt * 1e12;

    char caption[160];
    std::snprintf(caption, sizeof(caption),
        "ring lags the ideal by %+.1f ps (%+.1f tau), %s", p.lag_ps,
        p.lag * dt / ring.round_trip_time(),
        align ? "ideal shifted onto the ring" : "shown unshifted");
    p.caption = std::string(caption);
    std::cout << p.caption << std::endl;

    if (align) {
        p.power_diff = shift_samples(p.power_diff, p.lag);
        p.diff_real_norm = shift_samples(p.diff_real_norm, p.lag);
    }

    return p;
}

void Simulation::input_signal(const Propagation &p) {
    plt::figure_size(900, 380);
    plt::title("Input signal, " + p.ring_label);
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.in_norm),
        {{"color", "black"}, {"linewidth", "2"}, {"label", "input"}});
    plt::xlim(-2.5, 2.5);
    finish_axes("Time [ns]", "Input signal x(t)");
    plt::tight_layout();
}

void Simulation::waveforms(const Propagation &p) {
    plt::figure_size(900, 420);
    plt::title("Output waveform - " + p.caption);
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.diff_real_norm),
        {{"color", "black"}, {"linewidth", "2"},
        {"label", "ideal derivative"}});
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.ring_real_norm),
        {{"color", "red"}, {"linewidth", "2"},
        {"label", "MRR output (real)"}});
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.ring_imag_norm),
        {{"color", "red"}, {"linestyle", "--"}, {"linewidth", "2"},
        {"label", "MRR output (imag)"}});
    plt::xlim(-2.5, 2.5);
    finish_axes("Time [ns]", "Derivative y(t)");
    plt::tight_layout();
}

void Simulation::optical_power(const Propagation &p) {
    plt::figure_size(900, 420);
    plt::title("Optical power - " + p.caption);
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.power_diff),
        {{"color", "black"}, {"linewidth", "2"},
        {"label", "ideal derivative (power)"}});
    plt::plot(to_std_vec(p.time_ns), to_std_vec(p.power_ring),
        {{"color", "red"}, {"linewidth", "2"},
        {"label", "MRR output (power)"}});
    plt::xlim(-2.5, 2.5);
    finish_axes("Time [ns]", "|y(t)|^2");
    plt::tight_layout();
}
