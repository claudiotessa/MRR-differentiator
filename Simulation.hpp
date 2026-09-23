#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "MRR.hpp"

/**
 * @brief Numerical experiments on a ring: propagation, alignment and figures.
 */
class Simulation {

  public:
    // --- Frequency domain ------------------------------------------------

    /**
     * @brief One figure: magnitude and phase of a ring against the ideal
     *        n-th order differentiator, side by side.
     * @param `m` the ring to draw.
     * @param `n` order of the ideal differentiator to compare against.
     * @param `B` bandwidth the ring was designed for [Hz]; the magnitudes are
     *        normalised at B/2 and that edge is marked on the plot.
     * @param `heading` first line of the figure title.
     */
    static void response(const MRR &m, double n, double B,
                         const std::string &heading);

    /// One figure: the first order ring (n = 1), magnitude and phase.
    static void first_order_response(const MRR &m, double B);

    /// One figure: the fractional order ring, magnitude and phase.
    static void fractional_response(const MRR &m, double n, double B);


    // --- Time domain -----------------------------------------------------

    /**
     * @brief The optical field launched into the ring.
     *
     * Measured against the paper's error metric, the same ring scores ~5% on
     * the paper's Gaussian and ~40% on a 12th-order super-Gaussian.
     */
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
         * @brief A Gaussian whose spectrum sits at `ratio` times the ring's
         *        Eq. (4) width - the knob the paper's design rule turns.
         */
        static Input gaussian_matched(const MRR &ring, double ratio = 2.17);

        /// Flat-topped pulse of half-width `T0` and order 2*`m`.
        static Input super_gaussian(double T0, int m = 6);

        /// Hyperbolic-secant pulse of half-width `T0`.
        static Input sech(double T0);

        /// Rectangular pulse of half-width `T0`, raised cosine edges of `T0`/20.
        static Input rectangular(double T0);

        /// Samples the field on the time axis `t` [s], peak normalised to 1.
        Eigen::ArrayXd sample(const Eigen::ArrayXd &t) const;

        /// Short description for a plot legend.
        std::string describe() const;
    };

    /**
     * @brief Everything the time-domain figures draw, computed once.
     *
     * The waveforms are already normalised for plotting; `caption` states the
     * measured lag and whether it was removed.
     */
    struct Propagation {
        Eigen::ArrayXd time_ns;         // time axis [ns]
        Eigen::ArrayXd in_norm;         // input signal
        Eigen::ArrayXd diff_real_norm;  // ideal derivative, real part
        Eigen::ArrayXd ring_real_norm;  // ring output, real part
        Eigen::ArrayXd ring_imag_norm;  // ring output, imaginary part
        Eigen::ArrayXd power_diff;      // ideal derivative, |y|^2
        Eigen::ArrayXd power_ring;      // ring output, |y|^2
        long lag = 0;                   // ring delay [samples]
        double lag_ps = 0.0;            // ring delay [ps]
        double view_ns = 0.0;           // half-width of a sensible x range [ns]
        std::string caption;            // lag, formatted for a subplot title
        std::string ring_label;         // the ring's legend label
        std::string input_label;        // the input's legend label
    };

    /**
     * @brief Propagates `in` through the ring and through the ideal
     *        differentiator. Draws nothing.
     * @param `ring` the ring to propagate through.
     * @param `in` the input pulse.
     * @param `n` order of the differentiator the ring was built for.
     * @param `align` when true the ideal waveform is slid onto the ring by the
     *        measured lag, which answers "is the derivative the right shape?".
     *        When false the waveforms are returned as computed, so the ring's
     *        real latency stays visible. The lag is measured either way.
     * @param `N` number of samples; the window is sized from the pulse.
     */
    static Propagation propagate(const MRR &ring, const Input &in, double n,
                                 bool align = true, long N = 100000);

    /// One figure: the input pulse.
    static void input_signal(const Propagation &p);

    /// One figure: ideal derivative vs the ring output, real and imaginary.
    static void waveforms(const Propagation &p);

    /// One figure: optical power |y(t)|^2 of both.
    static void optical_power(const Propagation &p);

    /// Blocking call that draws every figure built so far.
    static void show();

    // --- Signal helpers --------------------------------------------------

    template <typename Derived>
    static std::vector<double> to_std_vec(const Eigen::ArrayBase<Derived> &arr) {
        return std::vector<double>(arr.derived().data(),
                                   arr.derived().data() + arr.size());
    }

    /**
     * @brief Moves the zero frequency to the centre of the array. For an
     *        even-length array this is its own inverse, so the same call
     *        undoes the shift.
     */
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

    /**
     * @brief Lag in samples that best aligns `a` onto `b`, from the peak of
     *        their cross-correlation. This is how the reference paper compares
     *        the ring output with the ideal derivative.
     */
    static long best_lag(const Eigen::ArrayXd &a, const Eigen::ArrayXd &b);

    /// Shifts `arr` later in time by `k` samples, zero-filling the vacated end.
    static Eigen::ArrayXd shift_samples(const Eigen::ArrayXd &arr, long k);

    /**
     * @brief Magnitude in dB, normalised at the reference frequency `f_ref` [Hz].
     *
     * Both the ring and the ideal response must be anchored at the same
     * frequency, and that frequency has to lie inside the differentiator band.
     * Normalising each curve by its own maximum instead anchors them at the
     * edge of the plot window, where the ring is already transparent and no
     * longer differentiates: the two curves are then forced to agree exactly
     * where they physically cannot and to disagree near DC, where they
     * actually do agree.
     */
    static Eigen::ArrayXd to_dB(const Eigen::ArrayXd &mag,
                                const Eigen::ArrayXd &freq_hz, double f_ref);
};

#endif
