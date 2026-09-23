#ifndef MRR_HPP
#define MRR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

class MRR {

    using Complex = std::complex<double>;
    static constexpr double c = 2.99792458e8; // lightspeed

  public:

    /**
     * @brief Coupler convention: the transmission matrix of the bus/ring
     *        coupler is [[r, -j*t], [-j*t, r]], so `r` is the diagonal
     *        (self-coupling, the residual) and `t` is the cross-coupling.
     */
    MRR(double R,
        double r,
        double xi,
        double n_eff,
        double n_g = -1.0,
        double detuning = 0.0
        ): R(R), r(r), xi(xi), n_eff(n_eff), n_g(n_g > 0.0 ? n_g : n_eff),
          detuning(detuning), L_r(2.0 * M_PI * R) { tau = (this->n_g * L_r) / c; }

    /**
     * @brief Creates a first order differentiator in critical coupling.
     * @param `B` the desired 3 dB bandwidth of the differentiator [Hz].
     * @param `R` radius of the MRR [m].
     * @param `n_eff` effective refractive index (also used as group index).
     */
    static MRR first_order(double B, double R, double n_eff);

    /**
     * @brief Creates a fractional order differentiator, under-coupled (r > xi).
     * @param `n` the order of the differentiator, 0 < n <= 1.
     * @param `R` radius of the MRR [m].
     * @param `xi` single-pass amplitude transmission (round-trip loss).
     * @param `n_eff` effective refractive index (also used as group index).
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
        return (r - xi * exp_neg_j_theta) / (1.0 - r * xi * exp_neg_j_theta);
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
    double self_coupling() const { return r; }          // diagonal, `r`
    double cross_coupling() const { return std::sqrt(1.0 - r * r); } //antidiagonal, `-jt`
    double round_trip_loss() const { return xi; }
    double group_index() const { return n_g; }
    double round_trip_time() const { return tau; }   // [s]
    double fsr() const { return 1.0 / tau; }         // [Hz]

    /// Finesse = FSR / FWHM, from the Airy linewidth of Eq. (1).
    double finesse() const {
        return M_PI * std::sqrt(r * xi) / (1.0 - r * xi);
    }

    /// 3 dB linewidth of the resonance [Hz].
    double fwhm() const { return fsr() / finesse(); }

    // --- Formatting ----------------------------------------------------

    /// Full parameter list, on two lines, for a subplot title.
    std::string params_string() const;

    /// Short "MRR (r = ..., xi = ...)" label for a legend entry.
    std::string label() const;

  private:
    double R;   // Radius [m]
    double L_r; // Length (circumference)
    double r;   // self coupling: diagonal of the coupler matrix
    double xi;  // ring loss (single-pass amplitude transmission)
    double n_eff; // effective index of the waveguide mode
    double n_g;   // group index of the waveguide mode
    double detuning;
    double tau; // round trip time
};

#endif
