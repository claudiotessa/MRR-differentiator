#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "MRR.hpp"

// @brief Numerical experiments on a ring: propagation, alignment and figures.
class Simulation {

  public:
    // @brief The optical field launched into the ring.
    struct Input {
        enum Shape {
            Gaussian,      // exp(-(t/T0)^2), what the paper uses
            SuperGaussian, // exp(-(t/T0)^(2m)), flat-topped
            Sech,          // sech(t/T0), the soliton pulse
            Rectangular    // hard-edged, worst case for a narrow-band ring
        };

        Shape shape = Gaussian;
        double T0 = 1e-12; // half-width [s]
        int m = 6;         // super-Gaussian half-order, only used by that shape

        /// The paper's input: a Gaussian of half-width `T0`.
        static Input gaussian(double T0);

        /**
         * @brief A Gaussian whose amplitude-spectrum FWHM is `ratio` times the
         *        ring's usable band, MRR::usable_band().
         *
         * The paper's rule is that the input stays inside the ring's FWHM, so
         * `ratio` <= 1 is the safe choice and smaller is more conservative.
         * Sizing against Eq. (4) instead would be backwards: that width
         * shrinks as the ring gets better and is zero at critical coupling.
         */
        static Input gaussian_matched(const MRR &ring, double ratio = 1.0);

        /// Flat-topped pulse of half-width `T0` and order 2*`m`.
        static Input super_gaussian(double T0, int m = 6);

        /// Hyperbolic-secant pulse of half-width `T0`.
        static Input sech(double T0);

        /// Rectangular pulse of half-width `T0`, raised cosine edges of
        /// `T0`/20.
        static Input rectangular(double T0);

        /// Samples the field on the time axis `t` [s], peak normalised to 1.
        Eigen::ArrayXd sample(const Eigen::ArrayXd &t) const;

        /// Short description for a plot legend.
        std::string describe() const;
    };

    // @brief Everything the TIME-DOMAIN figures draw, computed once.
    struct Propagation {
        Eigen::ArrayXd time_ns;        // time axis [ns]
        Eigen::ArrayXd in_norm;        // input signal
        Eigen::ArrayXd diff_real_norm; // ideal derivative, real part
        Eigen::ArrayXd ring_real_norm; // ring output, real part
        Eigen::ArrayXd ring_imag_norm; // ring output, imaginary part
        Eigen::ArrayXd power_diff;     // ideal derivative, |y|^2
        Eigen::ArrayXd power_ring;     // ring output, |y|^2
        long lag = 0;                  // ring delay [samples]
        double lag_ps = 0.0;           // ring delay [ps]
        double error_Dn = 0.0;         // Eq. (3), see power_error()
        double view_ns = 0.0;          // half-width of a sensible x range [ns]
        std::string caption;           // lag, formatted for a subplot title
        std::string ring_label;        // the ring's legend label
        std::string input_label;       // the input's legend label
    };

    // @brief Binds the simulation to a ring and a target order.
    Simulation(const MRR &ring, double n, long N = 100000)
        : ring(ring), n(n), N(N), has_result(false) {}

    // Getters
    const MRR &get_ring() const { return ring; }
    double get_order() const { return n; }
    const Propagation &get_last_result() const { return last_propagation; }

    /// Silences run()'s stdout block. The Monte Carlo runs thousands of
    /// propagations and none of them want to narrate.
    Simulation &set_verbose(bool v) {
        verbose = v;
        return *this;
    }
    double get_error() const { return last_propagation.error_Dn; }

    // --- Error metric ----------------------------------------------------
    /**
     * @brief Eq. (3): D_n = int | |f_n|^2 - |g_n|^2 | dt / int |g_n|^2 dt,
     *        with `out` = |f_n|^2 (ring) and `ideal` = |g_n|^2.
     *
     * Relative, so dt cancels and the sums stand in for the integrals. Both
     * waveforms must be peak-normalised and time-aligned first: the ring is
     * lossy and answers a ringdown late, and neither is shape error.
     */
    static double power_error(const Eigen::ArrayXd &out,
                              const Eigen::ArrayXd &ideal);

    /**
     * @brief Propagates the pulse through the ring and stores the result.
     */
    const Propagation &run(const Input &in, bool align = true);

  private:
    MRR ring;
    double n; // differentiation order
    long N;
    Propagation last_propagation;
    bool has_result = false;
    bool verbose = true;

    // @brief Moves the zero frequency to the centre of the array
    template <typename Derived>
    static auto fftshift(const Eigen::DenseBase<Derived> &vec) {
        using Scalar = typename Derived::Scalar;
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime,
                      Derived::ColsAtCompileTime>
            out(vec.rows(), vec.cols());
        Eigen::Index n = vec.size();
        Eigen::Index mid = (n + 1) / 2;
        out.head(n - mid) = vec.tail(n - mid);
        out.tail(mid) = vec.head(mid);
        return out;
    }

    // @brief Lag in samples that best aligns `a` onto `b`
    static long best_lag(const Eigen::ArrayXd &a, const Eigen::ArrayXd &b);

    /// Shifts `arr` later in time by `k` samples, zero-filling the vacated end.
    static Eigen::ArrayXd shift_samples(const Eigen::ArrayXd &arr, long k);
};

#endif
