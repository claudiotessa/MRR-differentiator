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

    // Sintetizza N = ceil(n) anelli IDENTICI di ordine n/N, come la Sez. 3.B
    // di [LIU25] (1.44 = 0.72+0.72, 2.1 = 0.7+0.7+0.7). La Sez. 2 ammette
    // anche 1+0.44 o 1+1+0.1, con stadi diversi fra loro: non lo facciamo,
    // qui gli anelli sono tutti uguali.
    MRRCascade(double n, double R, double xi, double n_eff, double n_g,
               double df = 0.0);

    // Parametri di un singolo anello fabbricato
    struct StageParams {
        double R, r, xi, n_eff, n_g, df;
    };

    // Cascata perturbata anello per anello: ogni stadio ha il suo campione.
    // I getter aggregati riportano la media sugli stadi.
    static MRRCascade perturbed(const MRRCascade &nominal,
                                const std::vector<StageParams> &stage_params);

    // Tutti gli stadi identici (R_custom <= 0 usa il raggio nominale)
    static MRRCascade perturbed(const MRRCascade &nominal, double r, double xi,
                                double n_eff, double n_g, double df,
                                double R_custom = -1.0);

    // Ordine complessivo erogato: somma degli ordini realizzati dagli stadi,
    // che con stadi non piu' identici non e' N * order() del primo.
    double achieved_order() const {
        double sum = 0.0;
        for (const MRR &s : stages)
            sum += s.order();
        return sum;
    }

    // Risposta spettrale totale: H_tot(f) = PROD H_i(f)

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

    // Banda utile: intervallo in cui |H_tot| segue la legge di potenza
    // ideale |2*pi*f|^n entro `tol_dB`. Misurata, non stimata: con N stadi
    // lo scarto relativo di ogni anello si moltiplica, quindi la banda si
    // stringe, ma non con la formula dei filtri passa-banda in cascata.
    // E' una larghezza in ampiezza (|H|, 20*log10), la stessa convenzione
    // della FWHM d'ampiezza usata da gaussian_matched().
    double usable_band(double tol_dB = 1.0) const;

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
