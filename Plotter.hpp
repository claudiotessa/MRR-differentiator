#ifndef PLOTTER_HPP
#define PLOTTER_HPP

#include "MRR.hpp"
#include "Simulation.hpp"

#include <string>

class Plotter {
  public:
    // The three time-domain subplots, from data already computed.
    static void plot_time_domain(const Simulation::Propagation &p);

    // Ring frequency response: magnitude [dB] and phase.
    static void plot_frequency_response(const MRR &ring, double n,
                                        long N = 100000);

    // Show all figures
    static void show();

    // --- Figure files -----------------------------------------------------
    /// Where save() writes, created if missing. Defaults to "fig".
    static void set_output_dir(const std::string &dir);

    /// Writes the current figure to <dir>/<stem>.pdf and closes it, so the
    /// next plot call starts clean. Vector output: it is going into a slide.
    static void save(const std::string &stem);

    static void plot_all(const Simulation::Propagation &p, const MRR &ring,
                         double n, bool show_immediately = true);

    // Overload to extract results automatically
    static void plot_all(const Simulation &sim, bool show_immediately = true);

  private:
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
