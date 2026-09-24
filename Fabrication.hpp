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
 *          one 200 mm wafer, 248 nm DUV, IME. Also the layout-dependent
 *          correlation model: variation is spatially correlated, so two
 *          devices track each other the closer together they sit.
 * [ROB22]  arXiv:2205.11481 - FDE sensitivity of n_eff to width, 220 nm.
 * [IPSR24] 2024 Integrated Photonic Systems Roadmap, Si photonics, T1 & T5.
 */
namespace fab {

/// The reference device and every number [LIU25] reports for it.
namespace paper {

inline constexpr double wg_width = 400e-9;  // [m]
inline constexpr double wg_height = 220e-9; // [m]
inline constexpr double R = 1.9e-6;         // ring radius [m]
inline constexpr double n_g = 4.05;         // Lumerical
inline constexpr double n_eff = 2.25;       // not stated; only sets where the
                                            // resonance lands, not the shape

// Theory-guided design, Sec. 2: the pair the paper picks for n = 0.54.
// Their t is our r. This is the nominal device we build on.
inline constexpr double xi = 0.9428;
inline constexpr double r_n054 = 0.9568;

// FDTD-extracted values, Sec. 3. Device 1 is the same ring measured rather
// than designed, device 2 a second radius, multiring the 0.72 stage.
inline constexpr double xi_fdtd = 0.9493;      // device 1, R = 1.9 um
inline constexpr double gap_n054 = 172e-9;     // device 1
inline constexpr double R_dev2 = 1.5e-6;
inline constexpr double r_dev2 = 0.8711;
inline constexpr double xi_dev2 = 0.8323;
inline constexpr double r_multiring = 0.947;   // gap 160 nm, their n = 0.72
inline constexpr double xi_multiring = 0.9493;

// Cross-check with MRR::order_of() on every (t, xi) pair [LIU25] quotes:
//   design  (0.9568, 0.9428) -> n = 0.5395  as claimed
//   dev 2   (0.8711, 0.8323) -> n = 0.5398  as claimed
//   multi   (0.9470, 0.9493) -> t < xi, over-coupled, no solution for 0.72
// The first two confirm our inversion and their Eq. (2). The third cannot
// realise 0.72: it sits on the wrong branch, where the excursion is ~2 pi.

// Errors [LIU25] reports. D_accept is their own "less than 10%" bar.
inline constexpr double D_054_theory = 0.021; // Sec. 2, design pair
inline constexpr double D_054_fdtd = 0.0484;  // Sec. 3.A, device 1
inline constexpr double D_054_dev2 = 0.08;    // Sec. 3.A, device 2
inline constexpr double D_144_fdtd = 0.0405;  // Sec. 3.B, 0.72 + 0.72
inline constexpr double D_210_fdtd = 0.0154;  // Sec. 3.B, 0.7 + 0.7 + 0.7
inline constexpr double D_accept = 0.10;

// Inputs: Gaussian half-widths, exp(-(t/T0)^2). Sec. 2 and Sec. 3.A both use
// 3 ps, and so does the whole n = 0.1 -> 1.8 sweep of Fig. 17. Only the
// multi-ring cascades of Sec. 3.B widen it; 7 ps is stated for n = 1.44, and
// n = 2.1 shares its figure (Fig. 15) so it is taken to share the pulse.
inline constexpr double input_T0 = 3e-12;     // single ring, and Fig. 17
inline constexpr double input_T0_144 = 7e-12; // n = 1.44
inline constexpr double input_T0_210 = 7e-12; // n = 2.1, UNSTATED, see above

// Their quoted widths: 120 GHz for the differentiator, 100 GHz for the input.
// 120 GHz matches the notch measured at half depth in AMPLITUDE (123.5 GHz),
// not the power FWHM (203.5 GHz).
inline constexpr double band_quoted = 120e9;
inline constexpr double input_band_quoted = 100e9;

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

/// Ring-to-ring correlation inside one device, [LU17].
///
/// [LU17] builds a spatial correlation matrix over the layout and draws the
/// Monte Carlo through its Cholesky factor: the wafer-scale term is common to
/// everything nearby, and only a small residual is per-device. The published
/// intra-die handle is that width variation tracks pattern density within a
/// 200 um radius, so the decay is millimetre-scale.
///
/// The rings of a cascade sit tens of microns apart, four orders of magnitude
/// inside that, so they are near-perfectly correlated: one common geometry
/// error with a small independent residual on top.
namespace layout {

/// UNSOURCED: centre-to-centre spacing of the rings in a cascade. Tens of
/// microns is what routing a 1.9 um ring costs.
inline constexpr double ring_pitch = 50e-6; // [m]

/// UNSOURCED magnitude. [LU17] gives the millimetre scale, not a fitted
/// length. At this pitch anything from 1 to 10 mm leaves rho > 0.95.
inline constexpr double corr_length = 5e-3; // [m]

/// Correlation between two devices `d` apart, exp(-d/Lc). ~0.99 as set.
inline double rho(double d = ring_pitch, double Lc = corr_length) {
    return std::exp(-d / Lc);
}

} // namespace layout

/// Sensitivities: what a nanometre of geometry costs.
namespace sensitivity {

/// [ROB22], w = 400 nm, h = 220 nm, 1550 nm. Wider guides are far better
/// (2.5e-5 at 2000 nm), a lever we cannot use on a 1.9 um ring.
inline constexpr double dneff_dwidth = 3.0e-3 / 1e-9; // [1/m]

/// UNSOURCED, order of magnitude. Barely matters: sigma_height is 3x smaller.
inline constexpr double dneff_dheight = 2.0e-3 / 1e-9; // [1/m]

/// [LIU25] gap scan at R = 1.9 um: gap 160 nm -> t = 0.947, gap 172 nm ->
/// t = 0.9568. Two points, no error bars. The tightest tolerance here.
inline constexpr double dr_dgap = 8.17e-4 / 1e-9; // [1/m]

/// [LIU25]'s two FDTD rings: R = 1.9 um -> xi = 0.9493, R = 1.5 um ->
/// xi = 0.8323. A 400 nm secant, so it overestimates the slope at 1.9 um
/// where the curve flattens.
inline constexpr double dxi_dradius = 2.925e-4 / 1e-9; // [1/m]

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
// and dh, xi follows dR. Measured on the n = 0.54 ring, 300 trials, heater on
// unless stated:
//
//   correlated                 yield 66.3%   mean D_n  9.23%
//   independent                yield 68.3%   mean D_n  9.52%
//   correlated, no heater      yield  3.0%   mean D_n 43.01%
//   correlated, dxi/dw = 3e-4  yield 57.0%   mean D_n 10.70%
//
// Correlating r and xi changes nothing: their drivers dw and dR are
// independent. The real dependence, r to n_eff, is the one the heater erases.
// The unmeasured dxi/dw is what actually costs yield.

} // namespace fab

#endif // FABRICATION_HPP
