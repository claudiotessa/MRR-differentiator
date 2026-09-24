#ifndef PLOTTER_HPP
#define PLOTTER_HPP

#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Simulation.hpp"

#include <string>
#include <vector>

class Plotter {
  public:
    /// Matplotlib without a window, for runs that only save().
    static void headless();

    /// Input, field and power in one figure.
    static void plot_time_domain(const Simulation::Propagation &p);
    /// Input and field only, the first two panels of plot_time_domain().
    static void plot_time_fields(const Simulation::Propagation &p);
    /// Power only, the traces Eq. (3) integrates.
    static void plot_time_power(const Simulation::Propagation &p);
    /// Magnitude and phase against the ideal |2*pi*f|^n, over three usable
    /// bands either side of the resonance.
    static void plot_frequency_response(const MRRCascade &cascade,
                                        long N = 131072);
    static void show(); // draws every figure built so far

    static void plot_all(const Simulation::Propagation &p,
                         const MRRCascade &cascade,
                         bool show_immediately = true);

    /// Where save() writes, created if missing. Defaults to "fig".
    static void set_output_dir(const std::string &dir);

    /// Writes the current figure to <dir>/<stem>.pdf and closes it, so the
    /// next plot call starts clean. Vector output: it is going into a slide.
    static void save(const std::string &stem);

    /// Overload that pulls the last result straight off the simulation.
    static void plot_all(const Simulation &sim, bool show_immediately = true);

    static void plot_monte_carlo(const MonteCarlo::Result &res,
                                 double threshold = 10.0);

    // --- Presentation figures, fed by figures.cpp ----------------------

    /// Iso-order curves r - xi against xi, `n_bold` highlighted, with the
    /// design point marked.
    static void plot_locus_map(const std::vector<double> &orders,
                               double n_bold, double r_design,
                               double xi_design);

    /// D_n along one iso-order locus, fixed pulse and pulse scaled with the
    /// ring. x is r, as in [LIU25] Fig. 7.
    static void plot_dn_along_locus(const std::vector<double> &r,
                                    const std::vector<double> &D_fixed,
                                    const std::vector<double> &D_scaled,
                                    double n, double r_design);

    /// D_n against the order for one pulse, with the benchmark cases on top:
    /// ours with the paper's pulses, and what [LIU25] reports.
    static void plot_dn_vs_order(const std::vector<double> &n,
                                 const std::vector<double> &D, double T0_ps,
                                 const std::vector<double> &bench_n,
                                 const std::vector<double> &bench_ours,
                                 const std::vector<double> &bench_paper,
                                 double threshold = 10.0);

    /// Monte Carlo devices on the locus map, green if D_n <= threshold.
    static void plot_mc_scatter(const MonteCarlo::Result &res, double n,
                                double r_design, double xi_design,
                                double threshold = 10.0);

    /// Nominal D_n (top) and Monte Carlo yield (bottom) along a locus.
    static void plot_yield_along_locus(const std::vector<double> &r,
                                       const std::vector<double> &D_nominal,
                                       const std::vector<double> &yield,
                                       double n, double r_design,
                                       double threshold = 10.0);

    /// Yield with one tolerance switched on at a time.
    static void plot_tolerance_breakdown(const std::vector<std::string> &labels,
                                         const std::vector<double> &yield,
                                         double threshold = 10.0);

  private:
    static void panel_input(const Simulation::Propagation &p);
    static void panel_field(const Simulation::Propagation &p);
    static void panel_power(const Simulation::Propagation &p);

    /// Draws iso-order curves on log axes; the caller owns the figure.
    static void draw_iso_orders(const std::vector<double> &orders,
                                double n_bold, double xi_lo, double xi_hi);

    static void plot_ring_vs_ideal(const std::vector<double> &x,
                                   const std::vector<double> &ring_data,
                                   const std::vector<double> &ideal_data,
                                   const std::string &ring_label);

    static void finish_axes(const std::string &xlabel,
                            const std::string &ylabel);

    template <typename Derived>
    static std::vector<double>
    to_std_vec(const Eigen::ArrayBase<Derived> &arr) {
        return std::vector<double>(arr.derived().data(),
                                   arr.derived().data() + arr.size());
    }

    static Eigen::ArrayXd to_dB(const Eigen::ArrayXd &mag,
                                const Eigen::ArrayXd &freq_hz, double f_ref);
};

#endif
