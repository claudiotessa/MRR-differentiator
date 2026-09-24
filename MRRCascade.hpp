#ifndef MRR_CASCADE_HPP
#define MRR_CASCADE_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "MRR.hpp"

class MRRCascade {
  public:
    MRRCascade()
        : total_order(0.0), R(0.0), xi(0.0), n_eff(0.0), n_g(0.0), df(0.0) {}

    // Costruttore universale: sintetizza N = ceil(n) anelli identici di ordine
    // n/N
    MRRCascade(double n, double R, double xi, double n_eff, double n_g,
               double df = 0.0);

    // Permette di specificare facoltativamente un raggio perturbato R_custom
    // (se <= 0 usa quello nominale)
    static MRRCascade perturbed(const MRRCascade &nominal, double r, double xi,
                                double n_eff, double n_g, double df,
                                double R_custom = -1.0);

    // Ordine di derivazione complessivo effettivamente erogato dalla cascata
    double achieved_order() const {
        if (stages.empty())
            return 0.0;
        return static_cast<double>(stages.size()) * stages[0].order();
    } // Risposta spettrale totale: H_tot(f) = PROD H_i(f)

    template <typename Derived>
    Eigen::ArrayXcd compute_H(const Eigen::ArrayBase<Derived> &Df) const {
        if (stages.empty())
            return Eigen::ArrayXcd::Ones(Df.size());

        Eigen::ArrayXcd H_tot = stages[0].compute_H(Df);
        for (size_t i = 1; i < stages.size(); ++i) {
            H_tot *= stages[i].compute_H(Df);
        }
        return H_tot;
    }

    template <typename Derived>
    Eigen::ArrayXd compute_phase(const Eigen::ArrayBase<Derived> &Df) const {
        return compute_H(Df).arg();
    }

    // Getters
    double order() const { return total_order; }
    size_t num_stages() const { return stages.size(); }
    double radius() const { return R; }
    double round_trip_loss() const { return xi; }
    double mode_index() const { return n_eff; }
    double group_index() const { return n_g; }
    double resonance_offset() const { return df; }
    double self_coupling() const {
        return stages.empty() ? 0.0 : stages[0].self_coupling();
    }

    // Parametri dei singoli stadi identici
    double stage_order() const {
        return stages.empty() ? 0.0 : total_order / stages.size();
    }
    double stage_self_coupling() const {
        return stages.empty() ? 0.0 : stages[0].self_coupling();
    }
    double round_trip_time() const {
        return stages.empty() ? 0.0 : stages[0].round_trip_time();
    }

    // Banda utile aggregata (la larghezza notch si riduce all'aumentare degli
    // stadi)
    double usable_band() const;

    std::string description() const;
    std::string label() const;
    std::string params_string() const;

  private:
    double total_order;
    double R;
    double xi;
    double n_eff;
    double n_g;
    double df;
    std::vector<MRR> stages;
};

#endif // MRR_CASCADE_HPP
