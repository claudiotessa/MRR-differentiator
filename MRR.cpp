#include "MRR.hpp"
#include <cmath>
#include <stdexcept>

MRR MRR::first_order(double B, double R, double n_eff) {
    double L_r = 2.0 * M_PI * R;
    double tau = (n_eff * L_r) / c;

    double tau_c = 1.0 / (M_PI * B);
    double tau_n = tau_c / tau;

    double r = std::sqrt(tau_n / (1.0 + tau_n));
    double t = std::sqrt(1.0 - r * r);

    double xi = t; // Critical coupling
    return MRR(R, t, xi, n_eff);
}

MRR MRR::fractional_order(double n, double R, double t, double n_eff) {

    // Exact solution to Eq. (2)
    double K = std::pow(std::tan(n * M_PI / 2.0), 2);

    double Aq = K * t * t;
    double Bq = K * (1.0 + std::pow(t, 4)) + std::pow(1.0 - t * t, 2);
    double Cq = Aq;

    double discriminant = Bq * Bq - 4.0 * Aq * Cq;

    if (discriminant < 0.0) {
        throw std::runtime_error(
            "Impossibile calcolare xi: discriminante negativo (valore di t non "
            "ammissibile per questo ordine n)");
    }

    double X = (Bq - std::sqrt(discriminant)) / (2.0 * Aq);
    double xi = std::sqrt(X);

    return MRR(R, t, xi, n_eff);
}
