#include "MRR.hpp"
#include <cmath>
#include <stdexcept>

MRR MRR::first_order(double B, double R, double n_eff) {
    double L_r = 2.0 * M_PI * R;
    double tau = (n_eff * L_r) / c;

    double tau_c = 1.0 / (M_PI * B);
    double tau_n = tau_c / tau;

    double r = std::sqrt(tau_n / (1.0 + tau_n));
    double t = std::sqrt(1.0 - r*r);
    double xi = t; // Critical coupling
                   //
    return MRR(R, t, xi, n_eff);
}

MRR MRR::fractional_order(double n, double R, double xi, double n_eff) {

    // if n < 1 ok, else call a cascade of equal MRR

    // Exact solution to Eq. (2)
    double K = std::pow(std::tan(n * M_PI / 2.0), 2);

    // Quadratic in Y = t^2, from Eq. (2) with xi fixed: Aq*Y^2 + Bq*Y + Cq = 0
    const double Aq = -xi * xi * (K + 1.0);
    const double Bq = K * std::pow(xi, 4) + K + 2.0 * xi * xi;
    const double Cq = Aq;    

    const double discriminant = Bq * Bq - 4.0 * Aq * Cq;

    if (discriminant < 0.0) {
        throw std::runtime_error("fractional_order: no real solution for t (discriminant < 0)");
    }

    // first root is the physical one (t in [xi,1]);
    // second root gives t > 1 (unphysical);
    const double Y = (-Bq + std::sqrt(discriminant)) / (2.0 * Aq);
    if (Y < xi * xi || Y > 1.0)
        throw std::runtime_error("fractional_order: computed t^2 outside valid domain [xi^2, 1]");

    const double t = std::sqrt(Y);
    return MRR(R, t, xi, n_eff);
}
