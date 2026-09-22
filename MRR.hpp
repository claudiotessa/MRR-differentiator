#ifndef MRR_HPP
#define MRR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdexcept>

class MRR {
    using Complex = std::complex<double>;
    static constexpr double c = 2.99792458e8;

  public:
    MRR(double R, double t, double xi, double n_eff, double n_g = -1.0,
        double detuning = 0.0)
        : R(R), t(t), xi(xi), n_eff(n_eff), n_g(n_g > 0.0 ? n_g : n_eff),
          detuning(detuning), L_r(2.0 * M_PI * R), tau((n_g * L_r) / c) {}

    /**
     * @brief Creates a first order differentiator in critical coupling.
     * @param `t` amplitude transmission coefficient.
     * @param `R` radius of the MRR [m].
     * @param `n_eff` effective refractive index.
     */
    static MRR first_order(double B, double R, double n_eff);

    /**
     * @brief Creates a fractional order differentiator in critical coupling.
     * @param `n` the order of the differentiator 0 < n < 1
     * @param `R` radius of the MRR [m].
     * @param `t` amplitude transmission coefficient.
     * @param `n_eff` effective refractive index.
     */
    static MRR fractional_order(double n, double R, double t, double n_eff);

    template <typename Derived>
    Eigen::ArrayXcd compute_H(const Eigen::ArrayBase<Derived> &Df) const {
        using namespace std::complex_literals;

        // theta = 2 * pi * tau * Df
        Eigen::ArrayXcd theta =
            (2.0 * M_PI * tau * Df.derived()).template cast<Complex>();
        Eigen::ArrayXcd exp_neg_j_theta = (-1.0i * theta).exp();

        // Equazione 1 del paper: Risposta della porta through
        return (t - xi * exp_neg_j_theta) / (1.0 - t * xi * exp_neg_j_theta);
    }

  private:
    double R;   // Radius
    double L_r; // Length (circumference)
    double t;
    double xi;
    double n_eff;
    double n_g;
    double detuning;
    double tau;
};

#endif
