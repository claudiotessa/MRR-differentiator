#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include <vector>

#include "MRR.hpp"
#include "Simulation.hpp"

/**
 * @brief Simulazione Monte Carlo per l'analisi di resa (yield) e tolleranza
 *        di fabbricazione del microring differenziatore.
 */
class MonteCarlo {
  public:
    struct Config {
        int trials = 500; // Numero di campioni estratti
        double lambda_0 =
            1550e-9; // Lunghezza d'onda centrale della portante [m]

        // Deviazioni standard (distribuzioni gaussiane attorno ai valori
        // nominali)
        double sigma_r =
            0.0015; // Fluttuazione gap accoppiatore (self-coupling)
        double sigma_xi = 0.002;  // Rugosità di parete / fluttuazione perdite
        double sigma_neff = 2e-4; // Tolleranza geometrica sull'indice di modo
        double sigma_ng = 0.02;   // Tolleranza sull'indice di gruppo

        // Eventuale sintonizzazione termica attiva (tuning residuo residuo in
        // GHz) Se attiva, sovrascrive o riduce il disallineamento puro da neff
        bool enable_thermal_tuning = false;
        double sigma_df_tuned =
            0.05e9; // Errore residuo del circuito termico [Hz]

        double yield_threshold = 0.10; // Soglia di resa: D_n <= 10%
        bool align_waveforms = true;   // Se misurare solo l'errore di forma
    };

    struct Result {
        std::vector<double> errors_Dn; // Errori D_n misurati (in percentuale)
        std::vector<double> r_samples;
        std::vector<double> xi_samples;
        std::vector<double> neff_samples;
        std::vector<double> ng_samples;
        std::vector<double> df_samples; // [GHz]

        double mean_error = 0.0;
        double std_error = 0.0;
        double median_error = 0.0;
        double max_error = 0.0;
        double yield_rate = 0.0; // Percentuale con D_n <= soglia

        void print_summary() const;
    };

    // Costruttore che usa la Configurazione di default
    MonteCarlo(const MRR &nominal_ring, double n,
               const Simulation::Input &pulse)
        : MonteCarlo(nominal_ring, n, pulse, Config()) {}

    // Costruttore con Configurazione personalizzata
    MonteCarlo(const MRR &nominal_ring, double n,
               const Simulation::Input &pulse, const Config &config)
        : nominal_ring(nominal_ring), n(n), pulse(pulse), config(config) {}

    /// Esegue la simulazione su tutti i campioni
    Result run(long sim_samples = 50000) const;

  private:
    MRR nominal_ring; // Theorically ideal MRR
    double n;
    Simulation::Input pulse;
    Config config;
};

#endif // MONTE_CARLO_HPP
