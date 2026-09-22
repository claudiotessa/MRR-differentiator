#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
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

int main() {
    // Griglia delle frequenze da -20 GHz a +20 GHz
    long N = 10000;
    ArrayXcd Df =
        ArrayXd::LinSpaced(N, -20e9, 20e9).cast<std::complex<double>>();
    ArrayXd freq_ghz = Df.real() / 1e9;

    // =========================================================
    // TEST 1: PRIMO ORDINE (n = 1)
    // =========================================================
    MRR mrr_1 = MRR::first_order(10e9, 100e-6, 2.4);

    ArrayXcd H_mrr_1 = mrr_1.compute_H(Df);
    ArrayXd H_mrr_1_dB =
        10.0 * ((H_mrr_1.abs() / H_mrr_1.abs().maxCoeff()).square()).log10();

    // Derivata Ideale: |2*pi*f|^1
    ArrayXd H_ideal_1_abs = (2.0 * M_PI * Df.real()).abs();
    ArrayXd H_ideal_1_dB =
        10.0 * ((H_ideal_1_abs / H_ideal_1_abs.maxCoeff()).square()).log10();

    // =========================================================
    // TEST 2: FRAZIONARIO (n = 0.54)
    // =========================================================
    // Nota: usando il tuo ordine parametri (n, R, t)
    MRR mrr_frac = MRR::fractional_order(0.54, 100e-6, 0.95, 2.4);

    ArrayXcd H_mrr_frac = mrr_frac.compute_H(Df);
    ArrayXd H_mrr_frac_dB =
        10.0 *
        ((H_mrr_frac.abs() / H_mrr_frac.abs().maxCoeff()).square()).log10();

    // Derivata Ideale: |2*pi*f|^0.54
    ArrayXd H_ideal_frac_abs = (2.0 * M_PI * Df.real()).abs().pow(0.54);
    ArrayXd H_ideal_frac_dB =
        10.0 *
        ((H_ideal_frac_abs / H_ideal_frac_abs.maxCoeff()).square()).log10();

    // =========================================================
    // PLOTTING
    // =========================================================
    plt::figure_size(900, 700);

    plt::subplot(2, 1, 1);
    plt::title("Primo Ordine (n = 1, Banda = 10 GHz)");
    plt::plot(
        to_std_vec(freq_ghz), to_std_vec(H_mrr_1_dB),
        {{"color", "red"}, {"linewidth", "3"}, {"label", "MRR (Fisico)"}});
    plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_1_dB),
              {{"color", "blue"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "Ideale"}});
    plt::ylim(-30.0, 0.0);
    plt::ylabel("Magnitudo [dB]");
    plt::grid(true);
    plt::legend();

    plt::subplot(2, 1, 2);
    plt::title("Frazionario (n = 0.54, t = 0.90)");
    plt::plot(
        to_std_vec(freq_ghz), to_std_vec(H_mrr_frac_dB),
        {{"color", "red"}, {"linewidth", "3"}, {"label", "MRR (Fisico)"}});
    plt::plot(to_std_vec(freq_ghz), to_std_vec(H_ideal_frac_dB),
              {{"color", "blue"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "Ideale"}});
    plt::ylim(-30.0, 0.0);
    plt::xlabel("Frequenza [GHz]");
    plt::ylabel("Magnitudo [dB]");
    plt::grid(true);
    plt::legend();

    plt::tight_layout();
    plt::show();

    return 0;
}
