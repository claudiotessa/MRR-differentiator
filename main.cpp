#include <iomanip>
#include <iostream>

#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"

int main() {
    // ---------------------------------------------------------------------
    // 1. Parametri fisici del differenziatore a singolo stadio (n = 0.54)
    // ---------------------------------------------------------------------
    const double n = 0.54; // Ordine di derivazione frazionaria (singolo anello)
    const double R = 1.9e-6;   // Raggio del risonatore: 1.9 um
    const double xi = 0.9428;  // Perdita round-trip nominale (dal paper)
    const double n_eff = 2.45; // Indice effettivo di modo
    const double n_g = 4.05;   // Indice di gruppo (guida Si 400x220 nm)

    // Istanziazione tramite la nuova MRRCascade:
    // con n = 0.54 alloca internamente un singolo stadio frazionario
    MRRCascade cascade(n, R, xi, n_eff, n_g);

    std::cout << "=== CONFIGURAZIONE NOMINALE ===" << std::endl;
    std::cout << cascade.params_string() << std::endl;

    // ---------------------------------------------------------------------
    // 2. Definizione impulso e simulazione nominale
    // ---------------------------------------------------------------------
    Simulation::Input pulse = Simulation::Input::gaussian_matched(cascade, 1.0);
    std::cout << "Impulso: " << pulse.describe() << std::endl;

    Simulation nominal_sim(cascade, 100000);
    const auto &nominal_res = nominal_sim.run(pulse, true, true);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Errore nominale Dn: " << nominal_res.error_Dn * 100.0
              << " %\n"
              << std::endl;

    // Visualizzazione forme d'onda e spettro nominale (tempo e frequenza)
    Plotter::plot_all(nominal_sim, false);

    // ---------------------------------------------------------------------
    // 3. Studio Monte Carlo (Configurazione tolleranze di fabbricazione)
    // ---------------------------------------------------------------------
    MonteCarlo::Config cfg;
    cfg.trials = 1000;          // Dispositivi estratti
    cfg.lambda_0 = 1550e-9;     // Portante ottica [m]
    cfg.sigma_r = 0.0015;       // Tolleranza gap di accoppiamento
    cfg.sigma_xi = 0.002;       // Tolleranza rugosità di parete / perdite
    cfg.sigma_neff = 2e-4;      // Tolleranza geometrica sull'indice di modo
    cfg.sigma_ng = 0.02;        // Dispersione
    cfg.yield_threshold = 0.10; // Resa accettata per Dn <= 10%
    cfg.align_waveforms = true; // Calcolo forma al netto del ritardo puro
    cfg.seed = 42;

    // SCENARIO A: Dispositivo non sintonizzato (Untuned)
    cfg.enable_thermal_tuning = false;
    std::cout << "--> Esecuzione Monte Carlo UNTUNED (deriva "
                 "termica/geometrica libera)..."
              << std::endl;
    MonteCarlo mc_untuned(cascade, pulse, cfg);
    auto res_untuned = mc_untuned.run(50000);
    res_untuned.print_summary();

    // SCENARIO B: Sintonizzazione termica attiva (Tuned)
    cfg.enable_thermal_tuning = true;
    cfg.sigma_df_tuned =
        0.05e9; // Errore residuo del feedback di bloccaggio: 50 MHz
    std::cout
        << "--> Esecuzione Monte Carlo TUNED (con micro-riscaldatore attivo)..."
        << std::endl;
    MonteCarlo mc_tuned(cascade, pulse, cfg);
    auto res_tuned = mc_tuned.run(50000);
    res_tuned.print_summary();

    // Grafico dell'istogramma statistico della resa
    Plotter::plot_monte_carlo(res_tuned, cfg.yield_threshold * 100.0);
    Plotter::show();

    return 0;
}
