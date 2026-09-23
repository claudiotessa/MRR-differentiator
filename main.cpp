#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "MRR.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;
using namespace std::complex_literals;

template <typename Derived>
std::vector<double> to_std_vec(const Eigen::ArrayBase<Derived> &arr) {
  return std::vector<double>(arr.derived().data(),
      arr.derived().data() + arr.size());
}

// Ring parameters, formatted for a subplot title.
std::string ring_params(const MRR &m) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "R = %.0f um, n_g = %.2f, tau = %.2f ps, FSR = %.1f GHz\n"
      "t = %.4f, xi = %.4f, finesse = %.1f, FWHM = %.2f GHz",
      m.radius() * 1e6, m.group_index(), m.round_trip_time() * 1e12,
      m.fsr() / 1e9, m.self_coupling(), m.round_trip_loss(), m.finesse(),
      m.fwhm() / 1e9);
  return std::string(buf);
}

// Short label for the legend entry of the physical curve.
std::string ring_label(const MRR &m) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "MRR (t = %.4f, xi = %.4f)",
      m.self_coupling(), m.round_trip_loss());
  return std::string(buf);
}

int main() {
  // Frequency grid from -20 GHz to +20 GHz
  long N = 10000;
  ArrayXcd Df =
    ArrayXd::LinSpaced(N, -20e9, 20e9).cast<std::complex<double>>();
  ArrayXd freq_ghz = Df.real() / 1e9;

  // =========================================================
  // TEST 1: FIRST ORDER (n = 1)
  // =========================================================
  MRR mrr_1 = MRR::first_order(10e9, 100e-6, 2.4);

  ArrayXcd H_mrr_1 = mrr_1.compute_H(Df);
  ArrayXd H_mrr_1_dB =
    10.0 * ((H_mrr_1.abs() / H_mrr_1.abs().maxCoeff()).square()).log10();
  ArrayXd H_mrr_1_phase = mrr_1.compute_phase(Df) / M_PI;

  // Ideal derivative: |2*pi*f|^1
  ArrayXd H_ideal_1_abs = (2.0 * M_PI * Df.real()).abs();
  ArrayXd H_ideal_1_dB =
    10.0 * ((H_ideal_1_abs / H_ideal_1_abs.maxCoeff()).square()).log10();
  // Phase of (j*2*pi*f)^1: +pi/2 for f > 0, -pi/2 for f < 0
  ArrayXd H_ideal_1_phase = Df.real().sign() * (1.0 / 2.0);

  // =========================================================
  // TEST 2: FRACTIONAL ORDER (n = 0.54)
  // =========================================================
  // Note: parameter order is (n, R, xi, n_eff)
  MRR mrr_frac = MRR::fractional_order(0.54, 100e-6, 0.90, 2.4);

  ArrayXcd H_mrr_frac = mrr_frac.compute_H(Df);
  ArrayXd H_mrr_frac_dB =
    10.0 *
    ((H_mrr_frac.abs() / H_mrr_frac.abs().maxCoeff()).square()).log10();
  ArrayXd H_mrr_frac_phase = mrr_frac.compute_phase(Df) / M_PI;

  // Ideal derivative: |2*pi*f|^0.54
  ArrayXd H_ideal_frac_abs = (2.0 * M_PI * Df.real()).abs().pow(0.54);
  ArrayXd H_ideal_frac_dB =
    10.0 *
    ((H_ideal_frac_abs / H_ideal_frac_abs.maxCoeff()).square()).log10();
  // Phase of (j*2*pi*f)^0.54: +0.54*pi/2 for f > 0, -0.54*pi/2 for f < 0
  ArrayXd H_ideal_frac_phase = Df.real().sign() * (0.54 / 2.0);

  // =========================================================
  // PLOTTING
  // =========================================================
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
  plt::ylim(-20.0, 0.0);
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
  plt::ylim(-20.0, 0.0);
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
  plt::show();

  return 0;
}
