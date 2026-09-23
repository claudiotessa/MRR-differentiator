#ifndef MRR_HPP
#define MRR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdexcept>

class MRR {

    using Complex = std::complex<double>;
    static constexpr double c = 2.99792458e8; // lightspeed

  public:

    MRR(double R,
        double t,
        double xi,
        double n_eff,
        double n_g = -1.0,
        double detuning = 0.0
        ): R(R), t(t), xi(xi), n_eff(n_eff), n_g(n_g > 0.0 ? n_g : n_eff),
          detuning(detuning), L_r(2.0 * M_PI * R) { tau = (this->n_g * L_r) / c; }

    /**
     * @brief Creates a first order differentiator in critical coupling.
     * @param `B` the desired 3db bandwidth [Hz]. FWHD
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
    static MRR fractional_order(double n, double R, double xi, double n_eff);

    template <typename Derived>
    Eigen::ArrayXcd compute_H(const Eigen::ArrayBase<Derived> &Df) const {
        using namespace std::complex_literals;

        // theta = 2 * pi * tau * Df
        Eigen::ArrayXcd theta =
            (2.0 * M_PI * tau * Df.derived()).template cast<Complex>();
        Eigen::ArrayXcd exp_neg_j_theta = (-1.0i * theta).exp();

        // Eq. (1) of the paper: through-port response
        return (t - xi * exp_neg_j_theta) / (1.0 - t * xi * exp_neg_j_theta);
    }

    /**
     * @brief Phase of the through-port response, wrapped in (-pi, pi].
     * @param `Df` frequency detuning from resonance [Hz].
     */
    template <typename Derived>
    Eigen::ArrayXd compute_phase(const Eigen::ArrayBase<Derived> &Df) const {
        return compute_H(Df).arg();
    }

    // --- Parameter accessors -------------------------------------------
    double radius() const { return R; }          // [m]
    double length() const { return L_r; }        // circumference [m]
    double self_coupling() const { return t; }
    double round_trip_loss() const { return xi; }
    double group_index() const { return n_g; }
    double round_trip_time() const { return tau; }   // [s]
    double fsr() const { return 1.0 / tau; }         // [Hz]

    /// Finesse = FSR / FWHM, from the Airy linewidth of Eq. (1).
    double finesse() const {
        return M_PI * std::sqrt(t * xi) / (1.0 - t * xi);
    }

    /// 3 dB linewidth of the resonance [Hz].
    double fwhm() const { return fsr() / finesse(); }

  private:
    double R;   // Radius
    double L_r; // Length (circumference)
    double t;   // self coupling
    double xi;  // ring loss
    double n_eff; // effective index of the waveguide mode
    double n_g;   // group index of the waveguide mode
    double detuning;
    double tau; // round trip time
};

#endif
