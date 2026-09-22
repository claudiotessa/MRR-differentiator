#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <unsupported/Eigen/FFT>

#include "MRR.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace Eigen;
using namespace std::complex_literals;

template <typename Derived>
auto fftshift(const Eigen::DenseBase<Derived> &vec) {
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime,
                  Derived::ColsAtCompileTime>
        out(vec.rows(), vec.cols());

    Eigen::Index n = vec.size();
    Eigen::Index mid = (n + 1) / 2; // matches MATLAB's ceil(n/2) split

    // Swap second half to first half, and first half to second half
    out.head(n - mid) = vec.tail(n - mid);
    out.tail(mid) = vec.head(mid);

    return out;
}

// Convert Eigen in std::vector for use in matplotlib-cpp
template <typename Derived>
std::vector<double> to_std_vec(const Eigen::ArrayBase<Derived> &arr) {
    return std::vector<double>(arr.derived().data(),
                               arr.derived().data() + arr.size());
}

int main() {

    long N = 1e5; // number of samples
    Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N, -10e-9, 10e-9);

    double dt = time(1) - time(0); // number of samples
    Eigen::ArrayXcd Df =
        Eigen::ArrayXd::LinSpaced(N, -1 / (2 * dt), 1 / (2 * dt))
            .cast<std::complex<double>>();

    double c = 3e8;
    double bandwidth = 10e9;

    double A = 1e10; // time scaling parameter
    ArrayXd E_in = (-((0.1 * A * time).pow(12))).exp(); // input signal

    double n_eff = 2.4;
    double R = 1e-4;
    double L_ring = 2.0 * M_PI * R;

    // explain here
    double tau = L_ring / (c / n_eff);
    double tau_c = 1.0 / (M_PI * bandwidth);
    double tau_n = tau_c / tau;

    // coupling coefficient
    double r = sqrt(tau_n / (1.0 + tau_n));
    double t = sqrt(1 - r * r);

    // give defs
    ArrayXcd beta = 2.0 * M_PI * Df * n_eff / c;
    double phi = 0.0;
    double gamma = r; // alpha + j * beta

    ArrayXcd phase =
        (-1.0i * beta * L_ring + ArrayXcd::Ones(beta.size()) * 1.0i * phi);
    ArrayXcd H_through =
        (r - gamma * phase.exp()) / (1.0 - r * gamma * phase.exp());
    ArrayXcd H_diff = 1.0i * r / (1.0 - r * r) * tau * (2.0 * M_PI * Df) *
                      (-1.0i * tau * M_PI * Df).exp();

    // FFT of the input
    FFT<double> fft;
    VectorXcd fft_raw;
    VectorXd E_in_vec = E_in.matrix();
    fft.fwd(fft_raw, E_in_vec);

    // align to Df
    VectorXcd in_ring = fftshift(fft_raw);

    VectorXcd out_ring_f = in_ring.array() * H_through;
    VectorXcd out_diff_f = in_ring.array() * H_diff;

    // IFFT
    VectorXcd out_ring_t, out_diff_t;
    fft.inv(out_ring_t, fftshift(out_ring_f));
    fft.inv(out_diff_t, fftshift(out_diff_f));

    ArrayXd power_ring = out_ring_t.array().abs2();
    ArrayXd power_diff = out_diff_t.array().abs2();

    std::cout << "Simulation completed" << std::endl;

    // Grandezze normalizzate per i grafici
    ArrayXd time_ns = time * 1e9;
    ArrayXd freq_ghz = Df.real() / 1e9;

    // 1. Spettri in dB (Figure 2 di MATLAB)[cite: 3, 4]
    ArrayXd abs_H = H_through.abs();
    ArrayXd abs_D = H_diff.abs();
    ArrayXd abs_IN = in_ring.array().abs();

    ArrayXd H_through_dB = 10.0 * ((abs_H / abs_H.maxCoeff()).square()).log10();
    ArrayXd H_diff_dB = 10.0 * (abs_D.square()).log10();
    ArrayXd IN_ring_dB = 10.0 * ((abs_IN / abs_IN.maxCoeff()).square()).log10();

    // 2. Forme d'onda temporali (Figure 1 di MATLAB)[cite: 3, 4]
    ArrayXd in_norm = E_in / E_in.maxCoeff();

    double max_diff = out_diff_t.real().array().maxCoeff();
    ArrayXd diff_real_norm = out_diff_t.real().array();
    if (std::abs(max_diff) > 1e-12) {
        diff_real_norm /= max_diff;
    }

    double max_ring = out_ring_t.real().array().maxCoeff();
    ArrayXd ring_real_norm = out_ring_t.real().array();
    ArrayXd ring_imag_norm = out_ring_t.imag().array();
    if (std::abs(max_ring) > 1e-12) {
        ring_real_norm /= max_ring;
        ring_imag_norm /= max_ring;
    }

    // Normalizzazione della potenza ottica[cite: 3, 4]
    if (power_diff.maxCoeff() > 1e-12)
        power_diff /= power_diff.maxCoeff();
    if (power_ring.maxCoeff() > 1e-12)
        power_ring /= power_ring.maxCoeff();

    double Mx = 4.0;
    double x_min = (-10e-9 * 1e9) / Mx;
    double x_max = (10e-9 * 1e9) / Mx;

    // ==========================================
    // FIGURA 1: Risposta Temporale (3 Subplot)[cite: 3, 4]
    // ==========================================
    plt::figure_size(850, 850);

    // Subplot 311: Segnale di ingresso[cite: 3, 4]
    plt::subplot(3, 1, 1);
    plt::plot(to_std_vec(time_ns), to_std_vec(in_norm),
              {{"color", "black"}, {"linewidth", "2"}, {"label", "input"}});
    plt::xlim(x_min, x_max);
    plt::xlabel("Time [ns]");
    plt::ylabel("Input signal y(t)");
    plt::grid(true);
    plt::legend();

    // Subplot 312: Confronto derivata ideale e uscita ottica[cite: 3, 4]
    plt::subplot(3, 1, 2);
    plt::plot(to_std_vec(time_ns), to_std_vec(diff_real_norm),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative"}});
    plt::plot(
        to_std_vec(time_ns), to_std_vec(ring_real_norm),
        {{"color", "red"}, {"linewidth", "2"}, {"label", "MRR output (real)"}});
    plt::plot(to_std_vec(time_ns), to_std_vec(ring_imag_norm),
              {{"color", "red"},
               {"linestyle", "--"},
               {"linewidth", "2"},
               {"label", "MRR output (imag)"}});
    plt::xlim(x_min, x_max);
    plt::xlabel("Time [ns]");
    plt::ylabel("Derivative y'(t)");
    plt::grid(true);
    plt::legend();

    // Subplot 313: Potenza ottica[cite: 3, 4]
    plt::subplot(3, 1, 3);
    plt::plot(to_std_vec(time_ns), to_std_vec(power_diff),
              {{"color", "black"},
               {"linewidth", "2"},
               {"label", "ideal derivative (power)"}});
    plt::plot(to_std_vec(time_ns), to_std_vec(power_ring),
              {{"color", "red"},
               {"linewidth", "2"},
               {"label", "MRR output (power)"}});
    plt::xlim(x_min, x_max);
    plt::xlabel("Time [ns]");
    plt::ylabel("| y'(t) |^2");
    plt::grid(true);
    plt::legend();

    // ==========================================
    // FIGURA 2: Risposta Spettrale in Frequenza (dB)[cite: 3, 4]
    // ==========================================
    plt::figure_size(850, 480);
    plt::plot(to_std_vec(freq_ghz), to_std_vec(H_through_dB),
              {{"color", "red"}, {"linewidth", "2"}, {"label", "MRR output"}});
    plt::plot(
        to_std_vec(freq_ghz), to_std_vec(H_diff_dB),
        {{"color", "blue"}, {"linewidth", "2"}, {"label", "ideal derivative"}});
    plt::plot(
        to_std_vec(freq_ghz), to_std_vec(IN_ring_dB),
        {{"color", "black"}, {"linewidth", "2"}, {"label", "input signal"}});
    plt::xlim(-10.0, 10.0);
    plt::ylim(-30.0, 0.0);
    plt::xlabel("Frequency [GHz]");
    plt::ylabel("Spectrum [dB]");
    plt::grid(true);
    plt::legend();

    plt::show();

    return 0;
}
