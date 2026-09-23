#include "MRR.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>

MRR MRR::first_order(double B, double R, double n_eff) {
    double L_r = 2.0 * M_PI * R;
    double tau = (n_eff * L_r) / c;

    double tau_c = 1.0 / (M_PI * B);
    double tau_n = tau_c / tau;

    // Self coupling (diagonal of the coupler matrix)
    // From the linearised Eq. (1): pi*tau*B = (1 - r^2) / r^2.
    double r = std::sqrt(tau_n / (1.0 + tau_n));
    double xi = r; // Critical coupling: xi = r

    return MRR(R, r, xi, n_eff);
}

MRR MRR::fractional_order(double n, double R, double xi, double n_eff) {
    // K = tan(n*pi/2)^2 is symmetric about n = 1, so n and 2 - n are
    // indistinguishable here: reject n > 1 and cascade rings instead.
    if (!(n > 0.0 && n <= 1.0))
        throw std::invalid_argument(
            "fractional_order: n must be in (0,1]; cascade rings for n > 1");

    // Exact solution to Eq. (2)
    double K = std::pow(std::tan(n * M_PI / 2.0), 2);

    // Quadratic in Y = r^2, from Eq. (2) with xi fixed: Aq*Y^2 + Bq*Y + Cq = 0
    const double Aq = -xi * xi * (K + 1.0);
    const double Bq = K * std::pow(xi, 4) + K + 2.0 * xi * xi;
    const double Cq = Aq;

    const double discriminant = Bq * Bq - 4.0 * Aq * Cq;

    if (discriminant < 0.0) {
        throw std::runtime_error(
            "fractional_order: no real solution for r (discriminant < 0)");
    }

    // Cq == Aq, so the two roots are Y and 1/Y: the first is the physical one
    // (r in [xi,1], i.e. under-coupled), the second gives r > 1 (unphysical).
    const double Y = (-Bq + std::sqrt(discriminant)) / (2.0 * Aq);
    if (Y < xi * xi || Y > 1.0)
        throw std::runtime_error(
            "fractional_order: computed r^2 outside valid domain [xi^2, 1]");

    const double r = std::sqrt(Y);
    return MRR(R, r, xi, n_eff);
}

std::string MRR::params_string() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "R = %.0f um, n_g = %.2f, tau = %.2f ps, FSR = %.1f GHz\n"
                  "r = %.4f, xi = %.4f, finesse = %.1f, linewidth = %.2f GHz, "
                  "dv = %.3f GHz",
                  radius() * 1e6, group_index(), round_trip_time() * 1e12,
                  fsr() / 1e9, self_coupling(), round_trip_loss(), finesse(),
                  fwhm() / 1e9, transition_width() / 1e9);
    return std::string(buf);
}

std::string MRR::label() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "MRR (r = %.4f, xi = %.4f)",
                  self_coupling(), round_trip_loss());
    return std::string(buf);
}
