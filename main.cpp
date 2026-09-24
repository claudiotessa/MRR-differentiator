#include "MRR.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"
#include <iostream>

// Waveguide parameters shared by every figure.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // mode index, 220 nm SOI strip
static const double n_g = 4.2;       // group index, same waveguide

// int main() {
//     const double n = 0.54; // fractional order
//
//     MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff, n_g);
//
//     // The paper drives its 0.54-order device with a Gaussian sized against
//     the
//     // ring's Eq. (4) width. Other shapes are available - super_gaussian(),
//     // sech(), rectangular() - but the Gaussian is the one the paper's error
//     // figures are measured with, so it is the only fair comparison.
//     Simulation sim(ring, n);
//
//     // Create the impulse coupled to the ring
//     Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);
//
//     sim.run(pulse, true);
//     Plotter::plot_all(sim);
//
//     return 0;
// }

int main() {
    const double n = 0.54; // Ordine frazionario target[cite: 11]

    // 1. Dispositivo nominale (Golden Device)
    MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff, n_g);
    Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);

    // 2. Verifica del caso ideale
    std::cout << ">>> Esecuzione del caso nominale..." << std::endl;
    Simulation nominal_sim(ring, n);
    nominal_sim.run(pulse, true);
    std::cout << "Errore Dn nominale: " << nominal_sim.get_error() * 100.0
              << " %\n"
              << std::endl;

    // 3. Configurazione dell'analisi di resa Monte Carlo
    MonteCarlo::Config config;
    config.trials = 500;           // Numero di chip simulati
    config.yield_threshold = 0.10; // Soglia di successo: Dn <= 10%
    config.align_waveforms =
        true; // Isola l'errore di forma (esclude la latenza)

    // Tolleranze di fabbricazione tipiche per SOI 220 nm
    config.sigma_r = 0.0015;    // Fluttuazione accoppiamento (gap litografico)
    config.sigma_xi = 0.0020;   // Fluttuazione perdite di propagazione
    config.sigma_neff = 1.0e-4; // Variazione geometrica su neff
    config.sigma_ng = 0.02;     // Variazione su ng

    // NOTA SULLA SINTONIZZAZIONE TERMICA:
    // Senza tuning (false), una sigma_neff di 1e-4 sposta la risonanza di ~4
    // GHz, portando la resa a ~0%. Con tuning attivo (true), un
    // micro-riscaldatore riaggancia la portante lasciando solo un
    // disallineamento residuo.
    config.enable_thermal_tuning = true;
    config.sigma_df_tuned = 0.05e9; // Errore residuo del lock termico: 50 MHz

    // 4. Esecuzione Monte Carlo
    MonteCarlo mc(ring, n, pulse, config);
    MonteCarlo::Result results =
        mc.run(30000); // 30k punti FFT per velocizzare i 500 trial

    // 5. Stampa report a terminale
    results.print_summary();

    // 6. Visualizzazione grafica
    Plotter::plot_all(nominal_sim,
                      false); // Figure nominali (input, tempo, spettro)
    // TODO: implement this function
    // Plotter::plot_monte_carlo(results, config.yield_threshold *
    // 100.0); // Istogramma di resa
    Plotter::show();

    return 0;
}
