#ifndef FABRICATION_HPP
#define FABRICATION_HPP

#include <cmath>

/**
 * @brief Fabrication data for the Monte Carlo: geometry tolerances times
 *        sensitivities, each tagged with its source. Anything not sourced
 *        says UNSOURCED.
 *
 * [LIU25]  Liu et al., Appl. Opt. 64(7), 1625 (2025) - the paper this
 *          implements, docs/4.10*.pdf. Its own references are all
 *          differentiator theory and report no fabrication statistics.
 * [LU17]   Lu et al., Opt. Express 25(9), 9712 (2017) - 2074 racetracks on
 *          one 200 mm wafer, 248 nm DUV, IME.
 * [ROB22]  arXiv:2205.11481 - FDE sensitivity of n_eff to width, 220 nm.
 * [IPSR24] 2024 Integrated Photonic Systems Roadmap, Si photonics, T1 & T5.
 */
namespace fab {

/// The reference device, [LIU25] Section 3.A.
namespace paper {

inline constexpr double wg_width = 400e-9; // [m]
inline constexpr double R = 1.9e-6;        // ring radius [m]
inline constexpr double n_g = 4.05;        // Lumerical
inline constexpr double n_eff = 2.25;      // not stated; only sets where the
                                           // resonance lands, not the shape
inline constexpr double xi = 0.9483;       // bend-loss limited at this radius
inline constexpr double r_n054 = 0.9568;   // their n = 0.54 point, gap 172 nm
                                           // (their t is our r)

// Checked with MRR::order_of() on every (t, xi) pair [LIU25] quotes:
//   R = 1.5 um, (0.8711, 0.8323) -> n = 0.5398  as claimed
//   R = 1.9 um, (0.9568, 0.9483) -> n = 0.6254  claimed 0.54
//   R = 1.9 um, (0.9470, 0.9493) -> t < xi, over-coupled, no solution
// The first validates our inversion; the other two are inconsistent with the
// paper's own Eq. (2). We use fractional_order(), which gives r = 0.9610.

inline constexpr double D_054_theory = 0.021; // 2.1 %
inline constexpr double D_054_fdtd = 0.0484;  // 4.84 %
inline constexpr double D_accept = 0.10;      // "within 10%", their own bar

inline constexpr double input_T0 = 3e-12; // 3 ps Gaussian

} // namespace paper

/// Geometry tolerances, 1 sigma.
namespace process {

inline constexpr double sigma_width = 3.855e-9;  // [LU17]
inline constexpr double sigma_height = 1.316e-9; // [LU17]

/// A uniform CD bias of +dw closes the gap by dw, so gap error is width error.
inline constexpr double sigma_gap = sigma_width;

/// UNSOURCED: the centreline comes from the drawn path, so its error is mask
/// placement rather than CD bias. Taken as the same order, independent of dw.
inline constexpr double sigma_radius = sigma_width;

// Cross-checks on sigma_width: [IPSR24] T5 gives 4% width uniformity in 2024
// (~2.7 nm as a sigma at 400 nm), [ROB22] says +/-10 nm is within 3 sigma for
// foundry wafer data. [IPSR24] T1 gives +/-0.5% of 220 nm for thickness.
// Loss spread is 1-0.2 dB/cm [IPSR24 T5], negligible over an 11.9 um ring.
inline constexpr double sigma_alpha_dB_per_cm = 0.4;

} // namespace process

/// Sensitivities: what a nanometre of geometry costs.
namespace sensitivity {

/// [ROB22], w = 400 nm, h = 220 nm, 1550 nm. Wider guides are far better
/// (2.5e-5 at 2000 nm), a lever we cannot use on a 1.9 um ring.
inline constexpr double dneff_dwidth = 3.0e-3 / 1e-9; // [1/m]

/// UNSOURCED, order of magnitude. Barely matters: sigma_height is 3x smaller.
inline constexpr double dneff_dheight = 2.0e-3 / 1e-9; // [1/m]

/// From [LIU25]'s own gap scan at R = 1.9 um: gap 160 nm -> t = 0.947,
/// gap 172 nm -> t = 0.9568. The tightest tolerance in the design.
inline constexpr double dr_dgap = 8.2e-4 / 1e-9; // [1/m]

/// From [LIU25]'s two rings on the same waveguide: R = 1.9 um -> xi = 0.9483,
/// R = 1.5 um -> xi = 0.8323. A secant, so it overestimates the slope at
/// 1.9 um, where the curve flattens.
inline constexpr double dxi_dradius = 2.9e-4 / 1e-9; // [1/m]

/// Bend loss against width. UNMEASURED, so zero rather than invented. A
/// wider guide would raise xi while the same +dw lowers r, making the two
/// anti-correlated; MonteCarlo::Config::dxi_dwidth sweeps it.
inline constexpr double dxi_dwidth = 0.0; // [1/m]

/// UNSOURCED. Only sets the band, not the order.
inline constexpr double dng_dwidth = 0.0; // [1/m]

} // namespace sensitivity

// --- Derived sigmas ---------------------------------------------------

/// ~3.2e-3, the dominant tolerance.
inline double sigma_r() { return sensitivity::dr_dgap * process::sigma_gap; }

/// ~1.2e-2, dominated by width.
inline double sigma_neff() {
    return std::hypot(sensitivity::dneff_dwidth * process::sigma_width,
                      sensitivity::dneff_dheight * process::sigma_height);
}

/// Loss scatter from propagation loss alone over a round trip of `L` metres.
/// ~5e-5 here and irrelevant: at R = 1.9 um, dxi_dradius is what moves xi.
inline double sigma_xi_from_propagation(double xi, double L) {
    return xi * (std::log(10.0) / 20.0) *
           process::sigma_alpha_dB_per_cm * (L * 100.0);
}

/// Resonance offset with nothing re-locking it: df/f = -dn_eff/n_g. ~570 GHz
/// against a 6.2 THz FSR, which is the argument for a heater.
inline double sigma_df_untuned(double n_g, double lambda0 = 1550e-9) {
    return (2.99792458e8 / lambda0) * sigma_neff() / n_g;
}

/// UNSOURCED by the process references, which say nothing about control
/// loops. 10-100 MHz is the usual figure for a locked ring.
inline constexpr double sigma_df_tuned = 50e6; // [Hz]

// --- What the correlated draw is worth --------------------------------
// MonteCarlo draws dw, dh, dR once per device. r follows dw, n_eff follows dw
// and dh, xi follows dR. Measured over 300 trials, tuning on unless stated:
//
//   correlated                 yield 65.7%   sd(r-xi) 3.37e-3
//   independent                yield 66.0%   sd(r-xi) 3.62e-3
//   correlated, no heater      yield  2.7%
//   correlated, dxi/dw = 3e-4  yield 57.0%   sd(r-xi) 4.33e-3
//
// Correlating r and xi changes nothing: their drivers dw and dR are
// independent. The real dependence, r to n_eff, is the one the heater erases.
// The unmeasured dxi/dw is what actually costs yield.

} // namespace fab

#endif // FABRICATION_HPP
