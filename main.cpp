#include <iomanip>
#include <iostream>
#include <vector>

#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"

// Parametri fisici nominali (piattaforma SOI standard 220 nm)
static const double R_ring = 100e-6; // Raggio dell'anello: 100 um
static const double xi =
    0.99; // Trasmissione monorimbalzo (perdite di curvatura/guida)
static const double n_eff = 2.40; // Indice efficace di modo
static const double n_g = 4.20;   // Indice di gruppo

int main() {
    // =========================================================================
    // 1. BENCHMARK RAPIDO SUGLI ORDINI DEL PAPER (Tabella 1)
    // =========================================================================
    const std::vector<double> paper_orders = {0.54, 1.44, 2.10};

    std::cout
        << "===============================================================\n";
    std::cout
        << "        TEST BENCHMARK SUI CASI DEL PAPER (TABELLA 1)          \n";
    std::cout
        << "===============================================================\n";
    std::cout << std::left << std::setw(10) << "Ordine n" << std::setw(12)
              << "Stadi (N)" << std::setw(14) << "n per stadio" << std::setw(16)
              << "Banda [GHz]" << std::setw(12) << "Errore Dn" << "\n";
    std::cout
        << "---------------------------------------------------------------\n";

    for (double order : paper_orders) {
        MRRCascade casc(order, R_ring, xi, n_eff, n_g);
        Simulation::Input in = Simulation::Input::gaussian_matched(casc);

        Simulation sim(casc);
        const auto &res = sim.run(in, true, false); // silente

        std::cout << std::left << std::setw(10) << std::fixed
                  << std::setprecision(2) << order << std::setw(12)
                  << casc.num_stages() << std::setw(14) << std::fixed
                  << std::setprecision(2) << casc.stage_order() << std::setw(16)
                  << std::fixed << std::setprecision(3)
                  << casc.usable_band() / 1e9 << std::setw(10) << std::fixed
                  << std::setprecision(2) << res.error_Dn * 100.0 << " %\n";
    }
    std::cout << "============================================================="
                 "==\n\n";

    // =========================================================================
    // 2. SIMULAZIONE NOMINALE DETTAGLIATA (Caso n = 1.44)
    // =========================================================================
    const double target_n = 1.44; // Due anelli uniformi da n_i = 0.72
    std::cout << ">>> Configurazione simulazione nominale per n = " << target_n
              << " ...\n";

    MRRCascade cascade(target_n, R_ring, xi, n_eff, n_g);
    Simulation::Input pulse = Simulation::Input::gaussian_matched(cascade);

    Simulation nominal_sim(cascade);
    nominal_sim.print_setup(pulse);

    const auto &nominal_res = nominal_sim.run(pulse, true, false);
    std::cout << "Risultato nominale: " << nominal_res.caption << "\n\n";

    // =========================================================================
    // 3. ANALISI STATISTICA MONTE CARLO SUL DISPOSITIVO PROGETTATO
    // =========================================================================
    std::cout << ">>> Avvio analisi di tolleranza Monte Carlo..." << std::endl;

    MonteCarlo::Config mc_cfg;
    mc_cfg.trials = 500;           // 500 chip virtuali estratti
    mc_cfg.yield_threshold = 0.10; // Criterio di successo: Dn <= 10%
    mc_cfg.align_waveforms =
        true; // Isola l'errore di distorsione della forma d'onda
    mc_cfg.enforce_under_coupled = true; // Vincola il regime frazionario r > xi

    // Modello di variabilità di processo SOI
    mc_cfg.correlated = false;  // Variazione indipendente dei parametri ottici
    mc_cfg.sigma_r = 0.0015;    // Fluttuazione accoppiamento (gap litografico)
    mc_cfg.sigma_xi = 0.0020;   // Fluttuazione perdite di propagazione
    mc_cfg.sigma_neff = 1.0e-4; // Variazione geometrica su neff
    mc_cfg.sigma_ng = 0.02;     // Variazione su ng

    // Circuito termico attivo (micro-heaters per riagganciare la risonanza)
    mc_cfg.enable_thermal_tuning = true;
    mc_cfg.sigma_df_tuned = 0.05e9; // Disallineamento residuo: 50 MHz

    MonteCarlo mc(cascade, pulse, mc_cfg);
    MonteCarlo::Result mc_res =
        mc.run(30000); // 30k punti FFT per bilanciare velocità e accuratezza

    // Stampa del report numerico a terminale
    mc_res.print_summary();

    // =========================================================================
    // 4. GENERAZIONE DEI GRAFICI
    // =========================================================================
    std::cout << ">>> Apertura grafici (chiudi le finestre per terminare)..."
              << std::endl;
    Plotter::plot_all(nominal_sim,
                      false); // Forme d'onda temporali e spettro dB/fase
    Plotter::plot_monte_carlo(mc_res, mc_cfg.yield_threshold *
                                          100.0); // Istogramma di resa
    Plotter::show();

    return 0;
}
