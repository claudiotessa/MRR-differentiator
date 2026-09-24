#ifndef MRR_HPP
#define MRR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
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
    /**
     * @param `n_eff` mode index: fixes where the resonance sits.
     * @param `n_g` group index: fixes tau, and with it FSR and the band.
     * @param `df` resonance offset [Hz], see compute_H().
     */
    MRR(double R, double r, double xi, double n_eff, double n_g,
        double df = 0.0)
        : R(R), L_r(2.0 * M_PI * R), r(r), xi(xi), n_eff(n_eff), n_g(n_g),
          df(df), tau(n_g * (2.0 * M_PI * R) / c) {}

    MRR()
        : R(0.0), L_r(0.0), r(0.0), xi(0.0), n_eff(0.0), n_g(0.0), df(0.0),
          tau(0.0) {}

    /**
     * @brief Creates a first-order differentiator: the critically coupled ring.
     *
     * Critical coupling *is* r == xi, so there is nothing to solve
     *
     * @param `R` radius of the MRR [m].
     * @param `xi` single-pass amplitude transmission (round-trip loss).
     */
    static MRR first_order(double R, double xi, double n_eff, double n_g,
                           double df = 0.0);

    /**
     * @brief Creates an n-th order differentiator, under-coupled (r > xi).
     *
     * This is the paper's design flow and the only one: `xi` is *given* - the
     * radius and the process fix the round-trip loss - and `r` is the single
     * quantity you draw on the mask, through the coupler gap. Eq. (2) solves
     * for it exactly instead of scanning, which is the paper's contribution.
     *
     * @param `n` the order of the differentiator, 0 < n <= 1. At n == 1 the
     *        solution degenerates to critical coupling, r == xi.
     * @param `R` radius of the MRR [m].
     * @param `xi` single-pass amplitude transmission (round-trip loss).
     */
    static MRR fractional_order(double n, double R, double xi, double n_eff,
                                double n_g, double df = 0.0);

    /**
     * @brief Through-port response, Eq. (1), at detunings `Df` [Hz] measured
     *        from the *nominal* resonance.
     *
     * A fabricated ring does not resonate where it was drawn to, so the whole
     * response slides by `df`: the round-trip phase is 2*pi*tau*(Df - df).
     */
    template <typename Derived>
    Eigen::ArrayXcd compute_H(const Eigen::ArrayBase<Derived> &Df) const {
        using namespace std::complex_literals;

        Eigen::ArrayXcd theta =
            (2.0 * M_PI * tau * (Df.derived() - df)).template cast<Complex>();
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
    double radius() const { return R; }        // [m]
    double length() const { return L_r; }      // circumference [m]
    double self_coupling() const { return r; } // diagonal, `r`
    double cross_coupling() const {
        return std::sqrt(1.0 - r * r);
    } // antidiagonal, `-jt`
    double round_trip_loss() const { return xi; }
    double mode_index() const { return n_eff; }
    double group_index() const { return n_g; }
    double resonance_offset() const { return df; } // [Hz]
    double round_trip_time() const { return tau; } // [s]
    double fsr() const { return 1.0 / tau; }       // [Hz]

    // --- Resonance position --------------------------
    // n_g sets how wide the resonances are, n_eff sets where they sit.
    // compute_H() works in detuning *from* resonance, so n_eff cannot appear
    // there; it enters by turning an index error into an offset `df`.

    /// How many wavelengths fit round the ring: m = n_eff * L / lambda.
    double mode_order(double lambda0 = 1.55e-6) const {
        return std::round(n_eff * L_r / lambda0);
    }

    /// Where the ring really resonates [m]. Only whole m are allowed, so this
    /// is generally not `lambda0`.
    double resonance_wavelength(double lambda0 = 1.55e-6) const {
        return n_eff * L_r / mode_order(lambda0);
    }

    /// Offset [Hz] from an n_eff error: df/f = -dn_eff/n_g. The group index,
    /// not n_eff: the guide is dispersive, so shifting the resonance also
    /// moves n_eff(lambda), and working that through leaves n_g.
    double detuning_from_index_error(double dn_eff,
                                     double lambda0 = 1.55e-6) const {
        return -(c / lambda0) * dn_eff / n_g;
    }

    /// Offset [Hz] from a measured resonance shift `dlambda` [m].
    static double detuning_from_wavelength_shift(double dlambda,
                                                 double lambda0 = 1.55e-6) {
        return -c * dlambda / (lambda0 * lambda0);
    }

    // --- Achieved order ------------------------------
    /**
     * @brief Eq. (2) read forwards: the order a device with this (`r`, `xi`)
     *        actually realises, n = (2/pi)*atan[xi(1-r^2)/sqrt(D)] with
     *        D = (r^2-xi^2)(1-r^2 xi^2).
     *
     * fractional_order() solves this for `r` given a target `n`; this is the
     * other direction, for a ring that came out of the process rather than off
     * the mask. Returns 1 at critical coupling and NaN when over-coupled
     * (r < xi), where Eq. (2) has no real solution and the device is a
     * different one, not a worse one.
     */
    static double order_of(double r, double xi) {
        const double d = (r * r - xi * xi) * (1.0 - r * r * xi * xi);
        if (d < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (d == 0.0)
            return 1.0; // atan(inf)
        return (2.0 / M_PI) *
               std::atan(xi * (1.0 - r * r) / std::sqrt(d));
    }

    /// The order this ring realises, see order_of().
    double order() const { return order_of(r, xi); }

    /// Finesse = FSR / FWHM, from the Airy linewidth of Eq. (1).
    double finesse() const { return M_PI * std::sqrt(r * xi) / (1.0 - r * xi); }

    /// 3 dB linewidth of the resonance [Hz], from the Airy lineshape.
    double fwhm() const { return fsr() / finesse(); }

    /**
     * @brief The differentiator's usable band [Hz]: the input spectrum has to
     * fit inside it.
     *
     * Note this is the standard *power* FWHM.
     */
    double usable_band() const { return fwhm(); }

    /// Power extinction at resonance [dB]. Set by the order alone.
    double extinction_dB() const {
        return 20.0 * std::log10(std::abs((r - xi) / (1.0 - r * xi)));
    }

    /**
     * @brief Width of the phase transition at resonance [Hz], Eq. (4).
     *
     * An imperfection metric, not a bandwidth: it measures how far the phase
     * response falls short of the ideal discontinuous n*pi step, so smaller is
     * better. It is exactly zero at critical coupling (r == xi), where the
     * jump is a true step - the ideal first-order differentiator. Do not size
     * an input pulse against it; use usable_band().
     */
    double phase_transition_width() const {
        const double a = xi * (1.0 + r * r) / (r * (1.0 + xi * xi));
        return std::acos(std::clamp(a, -1.0, 1.0)) / (tau * M_PI);
    }

    // --- Formatting ----------------------------------------------------

    /// Full parameter list, on two lines, for a subplot title.
    std::string params_string() const;

    /// Short "MRR (r = ..., xi = ...)" label for a legend entry.
    std::string label() const;

  private:
    double R;     // Radius [m]
    double L_r;   // Length (circumference)
    double r;     // self coupling: diagonal of the coupler matrix
    double xi;    // ring loss (single-pass amplitude transmission)
    double n_eff; // mode index: sets the resonance position
    double n_g;   // group index: sets tau
    double df;    // resonance offset from nominal [Hz]
    double tau;   // round trip time
};

#endif
