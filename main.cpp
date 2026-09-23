#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>
#include <unsupported/Eigen/FFT>
#include <vector>

#include "MRR.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;
using namespace std::complex_literals;

// Waveguide parameters shared by both figures.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // effective / group index

template <typename Derived>
std::vector<double> to_std_vec(const ArrayBase<Derived> &arr) {
  return std::vector<double>(arr.derived().data(),
      arr.derived().data() + arr.size());
}

// Moves the zero frequency to the centre of the array. For an even-length
// array this is its own inverse, so the same call undoes the shift.
template <typename Derived> auto fftshift(const DenseBase<Derived> &vec) {
  using Scalar = typename Derived::Scalar;
  Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> out(
      vec.rows(), vec.cols());
  Index n = vec.size();
  Index mid = (n + 1) / 2;
  out.head(n - mid) = vec.tail(n - mid);
  out.tail(mid) = vec.head(mid);
  return out;
}

// The ring output lags the ideal derivative. 
// The two waveforms are therefore aligned numerically, which is also how the
// reference paper compares them when it reports cross-correlation errors.
long best_lag(const ArrayXd &a, const ArrayXd &b) {
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

// Shifts `arr` later in time by `k` samples, zero-filling the vacated end.
ArrayXd shift_samples(const ArrayXd &arr, long k) {
  const long N = arr.size();
  ArrayXd out = ArrayXd::Zero(N);
  if (k >= 0)
    out.tail(N - k) = arr.head(N - k);
  else
    out.head(N + k) = arr.tail(N + k);
  return out;
}

// Magnitude in dB, normalised at the reference frequency `f_ref` [Hz].
//
// Both the ring and the ideal response must be anchored at the same frequency,
// and that frequency has to lie inside the differentiator band. Normalising
// each curve by its own maximum instead anchors them at the edge of the plot
// window, where the ring is already transparent and no longer differentiates:
// the two curves are then forced to agree exactly where they physically cannot
// and to disagree near DC, where they actually do agree.
ArrayXd to_dB(const ArrayXd &mag, const ArrayXd &freq_hz, double f_ref) {
  Eigen::Index i;
  ((freq_hz.abs() - f_ref).abs()).minCoeff(&i);
  return 20.0 * (mag / mag(i)).log10();
}

// Ring parameters, formatted for a subplot title.
std::string ring_params(const MRR &m) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "R = %.0f um, n_g = %.2f, tau = %.2f ps, FSR = %.1f GHz\n"
      "r = %.4f, xi = %.4f, finesse = %.1f, FWHM = %.2f GHz",
      m.radius() * 1e6, m.group_index(), m.round_trip_time() * 1e12,
      m.fsr() / 1e9, m.self_coupling(), m.round_trip_loss(), m.finesse(),
      m.fwhm() / 1e9);
  return std::string(buf);
}

// Short label for the legend entry of the physical curve.
std::string ring_label(const MRR &m) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "MRR (r = %.4f, xi = %.4f)",
      m.self_coupling(), m.round_trip_loss());
  return std::string(buf);
}

// =========================================================================
// FIGURE 1: FREQUENCY RESPONSE VS THE IDEAL DIFFERENTIATOR
// =========================================================================
void figure_frequency_response() {
  // Frequency grid from -20 GHz to +20 GHz
  long N = 10000;
  ArrayXcd Df =
    ArrayXd::LinSpaced(N, -20e9, 20e9).cast<std::complex<double>>();
  ArrayXd freq_hz = Df.real();
  ArrayXd freq_ghz = freq_hz / 1e9;

  // Target 3 dB bandwidth of the differentiators. The curves are normalised at
  // B/2, the edge of the band the rings are designed for.
  const double B = 10e9;
  const double f_ref = B / 2.0;

  // --- First order (n = 1) ---
  MRR mrr_1 = MRR::first_order(B, R_ring, n_eff);

  ArrayXcd H_mrr_1 = mrr_1.compute_H(Df);
  ArrayXd H_mrr_1_dB = to_dB(H_mrr_1.abs(), freq_hz, f_ref);
  ArrayXd H_mrr_1_phase = mrr_1.compute_phase(Df) / M_PI;

  // Ideal derivative: |2*pi*f|^1
  ArrayXd H_ideal_1_abs = (2.0 * M_PI * freq_hz).abs();
  ArrayXd H_ideal_1_dB = to_dB(H_ideal_1_abs, freq_hz, f_ref);
  // Phase of (j*2*pi*f)^1: +pi/2 for f > 0, -pi/2 for f < 0
  ArrayXd H_ideal_1_phase = freq_hz.sign() * (1.0 / 2.0);

  // --- Fractional order (n = 0.54) ---
  MRR mrr_frac = MRR::fractional_order(0.54, R_ring, 0.90, n_eff);

  ArrayXcd H_mrr_frac = mrr_frac.compute_H(Df);
  ArrayXd H_mrr_frac_dB = to_dB(H_mrr_frac.abs(), freq_hz, f_ref);
  ArrayXd H_mrr_frac_phase = mrr_frac.compute_phase(Df) / M_PI;

  // Ideal derivative: |2*pi*f|^0.54
  ArrayXd H_ideal_frac_abs = (2.0 * M_PI * freq_hz).abs().pow(0.54);
  ArrayXd H_ideal_frac_dB = to_dB(H_ideal_frac_abs, freq_hz, f_ref);
  // Phase of (j*2*pi*f)^0.54: +0.54*pi/2 for f > 0, -0.54*pi/2 for f < 0
  ArrayXd H_ideal_frac_phase = freq_hz.sign() * (0.54 / 2.0);

  plt::figure_size(1200, 700);

  // --- First order: magnitude ---
  plt::subplot(2, 2, 1);
  plt::title("First order (n = 1, target B = 10 GHz)\n" + ring_params(mrr_1));
  plt::plot(
      to_std_vec(freq_ghz), to_std_vec(H_mrr_1_dB),
      {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label(mrr_1)}});
  plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_1_dB),
      {{"color", "blue"},
      {"linestyle", "--"},
      {"linewidth", "2"},
      {"label", "Ideal"}});
  plt::axvline(f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
  plt::axvline(-f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
  plt::ylim(-25.0, 15.0);
  plt::ylabel("Magnitude [dB]");
  plt::grid(true);
  plt::legend();

  // --- First order: phase ---
  plt::subplot(2, 2, 2);
  plt::title("First order - phase (ideal = +/- 0.50 pi)");
  plt::plot(
      to_std_vec(freq_ghz), to_std_vec(H_mrr_1_phase),
      {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label(mrr_1)}});
  plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_1_phase),
      {{"color", "blue"},
      {"linestyle", "--"},
      {"linewidth", "2"},
      {"label", "Ideal"}});
  plt::ylim(-1.0, 1.0);
  plt::ylabel("Phase [rad / pi]");
  plt::grid(true);
  plt::legend();

  // --- Fractional order: magnitude ---
  plt::subplot(2, 2, 3);
  plt::title("Fractional order (n = 0.54)\n" + ring_params(mrr_frac));
  plt::plot(
      to_std_vec(freq_ghz), to_std_vec(H_mrr_frac_dB),
      {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label(mrr_frac)}});
  plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_frac_dB),
      {{"color", "blue"},
      {"linestyle", "--"},
      {"linewidth", "2"},
      {"label", "Ideal"}});
  plt::axvline(f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
  plt::axvline(-f_ref / 1e9, 0.0, 1.0, {{"color", "gray"}, {"linestyle", ":"}});
  plt::ylim(-25.0, 15.0);
  plt::xlabel("Frequency [GHz]");
  plt::ylabel("Magnitude [dB]");
  plt::grid(true);
  plt::legend();

  // --- Fractional order: phase ---
  plt::subplot(2, 2, 4);
  plt::title("Fractional order - phase (ideal = +/- 0.27 pi)");
  plt::plot(
      to_std_vec(freq_ghz), to_std_vec(H_mrr_frac_phase),
      {{"color", "red"}, {"linewidth", "3"}, {"label", ring_label(mrr_frac)}});
  plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_frac_phase),
      {{"color", "blue"},
      {"linestyle", "--"},
      {"linewidth", "2"},
      {"label", "Ideal"}});
  plt::ylim(-1.0, 1.0);
  plt::xlabel("Frequency [GHz]");
  plt::ylabel("Phase [rad / pi]");
  plt::grid(true);
  plt::legend();

  plt::tight_layout();
}

// =========================================================================
// FIGURE 2: TIME-DOMAIN PROPAGATION OF A SUPER-GAUSSIAN PULSE
// =========================================================================
void figure_time_domain() {
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

  // --- Fractional MRR built through our own factory ---
  const double n = 0.54;
  const double xi = 0.99;
  MRR ring = MRR::fractional_order(n, R_ring, xi, n_eff);

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

  // Optical power |y(t)|^2
  ArrayXd power_ring = out_ring_t.array().abs2();
  ArrayXd power_diff = out_diff_t.array().abs2();

  // --- Normalisation for plotting ---
  ArrayXd time_ns = time * 1e9;
  ArrayXd in_norm = E_in / E_in.maxCoeff();

  ArrayXd diff_real_norm = out_diff_t.real().array();
  double max_diff = diff_real_norm.maxCoeff();
  if (std::abs(max_diff) > 1e-12)
    diff_real_norm /= max_diff;

  ArrayXd ring_real_norm = out_ring_t.real().array();
  ArrayXd ring_imag_norm = out_ring_t.imag().array();
  double max_ring = ring_real_norm.maxCoeff();
  if (std::abs(max_ring) > 1e-12) {
    ring_real_norm /= max_ring;
    ring_imag_norm /= max_ring;
  }

  if (power_diff.maxCoeff() > 1e-12)
    power_diff /= power_diff.maxCoeff();
  if (power_ring.maxCoeff() > 1e-12)
    power_ring /= power_ring.maxCoeff();

  // The ideal differentiator has zero group delay, so the offset between the
  // two waveforms is a real property of the ring.
  // The lag is measured and reported, but the curves are plotted
  // unshifted so the ring's actual latency stays visible.
  const long lag = best_lag(power_diff, power_ring);
  const double lag_ps = lag * dt * 1e12;

  char align[160];
  std::snprintf(align, sizeof(align),
      "ring lags the ideal by %+.1f ps (%+.1f tau), shown unshifted", lag_ps,
      lag * dt / ring.round_trip_time());
  std::cout << align << std::endl;

  // Uncomment to slide the ideal onto the ring.
  power_diff = shift_samples(power_diff, lag);
  diff_real_norm = shift_samples(diff_real_norm, lag);

  plt::figure_size(900, 950);

  // --- Input signal ---
  plt::subplot(3, 1, 1);
  plt::title("Time domain, " + ring_label(ring) + ", n = 0.54");
  plt::plot(to_std_vec(time_ns), to_std_vec(in_norm),
      {{"color", "black"}, {"linewidth", "2"}, {"label", "input"}});
  plt::xlim(-2.5, 2.5);
  plt::xlabel("Time [ns]");
  plt::ylabel("Input signal x(t)");
  plt::grid(true);
  plt::legend();

  // --- Output waveforms, real and imaginary parts ---
  plt::subplot(3, 1, 2);
  plt::plot(to_std_vec(time_ns), to_std_vec(diff_real_norm),
      {{"color", "black"}, {"linewidth", "2"},
      {"label", "ideal derivative"}});
  plt::plot(to_std_vec(time_ns), to_std_vec(ring_real_norm),
      {{"color", "red"}, {"linewidth", "2"},
      {"label", "MRR output (real)"}});
  plt::plot(to_std_vec(time_ns), to_std_vec(ring_imag_norm),
      {{"color", "red"}, {"linestyle", "--"}, {"linewidth", "2"},
      {"label", "MRR output (imag)"}});
  plt::xlim(-2.5, 2.5);
  plt::title(std::string(align));
  plt::xlabel("Time [ns]");
  plt::ylabel("Derivative y(t)");
  plt::grid(true);
  plt::legend();

  // --- Optical power ---
  plt::subplot(3, 1, 3);
  plt::plot(to_std_vec(time_ns), to_std_vec(power_diff),
      {{"color", "black"}, {"linewidth", "2"},
      {"label", "ideal derivative (power)"}});
  plt::plot(to_std_vec(time_ns), to_std_vec(power_ring),
      {{"color", "red"}, {"linewidth", "2"},
      {"label", "MRR output (power)"}});
  plt::xlim(-2.5, 2.5);
  plt::xlabel("Time [ns]");
  plt::ylabel("|y(t)|^2");
  plt::grid(true);
  plt::legend();

  plt::tight_layout();
}

int main() {
  figure_time_domain();       // main result: the differentiated waveform
  figure_frequency_response(); // supporting: |H| and phase vs the ideal
  plt::show();
  return 0;
}
