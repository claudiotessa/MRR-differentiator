#include "Simulation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <unsupported/Eigen/FFT>

using namespace Eigen;

// =========================================================================
// SIGNAL HELPERS
// =========================================================================

// Aligns the curves so they can be compared
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

// =========================================================================
// INPUT PULSES
// =========================================================================

Simulation::Input Simulation::Input::gaussian(double T0) {
    Input in;
    in.shape = Gaussian;
    in.T0 = T0;
    return in;
}

Simulation::Input Simulation::Input::gaussian_matched(const MRRCascade &cascade,
                                                      double ratio) {
    // exp(-(t/T0)^2) transforms to exp(-(pi*f*T0)^2), whose amplitude FWHM is
    // 2*sqrt(ln2)/(pi*T0). Solve that for the wanted fraction of the band.
    const double band = cascade.usable_band();
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
        std::snprintf(buf, sizeof(buf), "Gaussian, T0 = %.3g ps", T0 * 1e12);
        break;
    case SuperGaussian:
        std::snprintf(buf, sizeof(buf),
                      "super-Gaussian (order %d), T0 = %.3g ps", 2 * m,
                      T0 * 1e12);
        break;
    case Sech:
        std::snprintf(buf, sizeof(buf), "sech, T0 = %.3g ps", T0 * 1e12);
        break;
    case Rectangular:
        std::snprintf(buf, sizeof(buf), "rectangular, T0 = %.3g ps", T0 * 1e12);
        break;
    }
    return std::string(buf);
}

// =========================================================================
// TIME-DOMAIN COMPUTATION
// =========================================================================
const Simulation::Propagation &Simulation::run(const Input &in, bool align,
                                               bool verbose) {
    if (N < 2) {
        throw std::invalid_argument("Simulation::run: N must be at least 2");
    }
    if (!std::isfinite(in.T0) || in.T0 <= 0.0) {
        throw std::invalid_argument(
            "Simulation::run: the pulse T0 must be finite and > 0");
    }

    // 1. Time window, long enough for the ring to have rung down and for the
    // slow t^-(n+1) tail of a fractional derivative not to wrap round: at
    // 10 T0 / 40 ringdowns D_0.54 was still 0.17 points off its limit.
    const double tau_stage = cascade.round_trip_time();
    const double r_stage = cascade.stage_self_coupling();
    const double xi_stage = cascade.round_trip_loss();

    const double ringdown = tau_stage / (1.0 - r_stage * xi_stage);
    const double window = std::max(40.0 * in.T0, 160.0 * ringdown);

    if (!std::isfinite(window) || window <= 0.0) {
        throw std::runtime_error(
            "Simulation::run: the time window is not finite");
    }

    // Time axis and sampling step.
    ArrayXd time = ArrayXd::LinSpaced(N, -window, window);
    const double dt = time(1) - time(0);

    // 2. Frequency grid centred on 0 Hz: the baseband spectrum around the
    // carrier.
    const long half = N / 2;
    ArrayXcd Df = ((ArrayXd::LinSpaced(N, 0.0, static_cast<double>(N - 1)) -
                    static_cast<double>(half)) /
                   (static_cast<double>(N) * dt))
                      .cast<std::complex<double>>();

    // 3. Input field.
    ArrayXd E_in = in.sample(time);

    // 4. Cascade response, H_tot(f) = PROD H_i(f).
    ArrayXcd H_through = cascade.compute_H(Df);

    // 5. Ideal operator, (j*2*pi*f)^n.
    ArrayXcd H_diff(N);
    for (long i = 0; i < N; ++i) {
        std::complex<double> j_omega(0.0, 2.0 * M_PI * Df(i).real());
        H_diff(i) = (std::abs(j_omega) < 1e-18) ? std::complex<double>(0.0, 0.0)
                                                : std::pow(j_omega, n);
    }

    // 6. Propagation through the spectrum.
    Eigen::FFT<double> fft;
    VectorXcd fft_raw;
    VectorXd E_in_vec = E_in.matrix();
    fft.fwd(fft_raw, E_in_vec);

    VectorXcd in_spectrum = fftshift(fft_raw);

    VectorXcd out_ring_f = in_spectrum.array() * H_through;
    VectorXcd out_diff_f = in_spectrum.array() * H_diff;

    VectorXcd out_ring_t, out_diff_t;
    fft.inv(out_ring_t, ifftshift(out_ring_f));
    fft.inv(out_diff_t, ifftshift(out_diff_f));

    // 7. Time-domain waveforms, peak-normalised.
    Propagation p;
    p.ring_label = cascade.label();
    p.input_label = in.describe();
    // Fixed +/- 20 ps window, so the n = 0.1 / 0.54 / 1.44 / 2.1 figures are
    // all on the same scale and the ring's ringdown tail stays visible past
    // the pulse. Widened for a pulse too long to fit.
    p.view_ps = std::max(view_half_ps, 2.5 * in.T0 * 1e12);

    p.power_ring = out_ring_t.array().abs2();
    p.power_diff = out_diff_t.array().abs2();

    p.time_ps = time * 1e12;
    p.in_norm = E_in / E_in.maxCoeff();

    // Normalise the field to unit amplitude, i.e. by max|y|, not by the
    // largest positive sample: for n > 1 the dominant excursion is the
    // negative lobe, and dividing by the smaller positive peak pushed the
    // curve past -1 instead of bounding it by +/-1.
    p.diff_real_norm = out_diff_t.real().array();
    const double max_diff = out_diff_t.array().abs().maxCoeff();
    if (max_diff > 1e-12) {
        p.diff_real_norm /= max_diff;
    }

    p.ring_real_norm = out_ring_t.real().array();
    p.ring_imag_norm = out_ring_t.imag().array();
    const double max_ring = out_ring_t.array().abs().maxCoeff();
    if (max_ring > 1e-12) {
        p.ring_real_norm /= max_ring;
        p.ring_imag_norm /= max_ring;
    }

    if (p.power_diff.maxCoeff() > 1e-12) {
        p.power_diff /= p.power_diff.maxCoeff();
    }
    if (p.power_ring.maxCoeff() > 1e-12) {
        p.power_ring /= p.power_ring.maxCoeff();
    }

    // 8. Group delay of the ring, then D_n on the aligned pair, Eq. (3).
    p.lag = best_lag(p.power_diff, p.power_ring);
    p.lag_ps = p.lag * dt * 1e12;

    const ArrayXd ideal_aligned = shift_samples(p.power_diff, p.lag);
    p.error_Dn = power_error(p.power_ring, ideal_aligned);

    char caption[224];
    std::snprintf(caption, sizeof(caption),
                  "%s lags ideal by %+.1f ps (%+.1f tau), %s\nD_n = %.2f %%",
                  cascade.description().c_str(), p.lag_ps,
                  p.lag * dt / tau_stage,
                  align ? "ideal shifted onto output" : "shown unshifted",
                  p.error_Dn * 100.0);
    p.caption = std::string(caption);

    if (verbose) {
        std::cout << p.caption << std::endl;
    }

    // Alignment is for the figures only; D_n above is already measured on
    // the overlapped pair.
    if (align) {
        p.power_diff = ideal_aligned;
        p.diff_real_norm = shift_samples(p.diff_real_norm, p.lag);
    }

    last_propagation = p;
    has_result = true;
    return last_propagation;
}

void Simulation::print_setup(const Input &in) const {
    std::cout << "=== Simulation Configuration ===\n"
              << cascade.params_string() << '\n'
              << "Input pulse:    " << in.describe() << '\n'
              << "================================" << std::endl;
}
