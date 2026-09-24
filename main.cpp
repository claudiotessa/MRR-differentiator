#include <iomanip>
#include <iostream>
#include <vector>

#include "Fabrication.hpp"
#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"

// Dispositivo di riferimento: [LIU25] Sez. 3.A, via Fabrication.hpp. Le
// tolleranze sono state ricavate su questa geometria, cambiarla le invalida.
static const double R_ring = fab::paper::R;     // 1.9 um
static const double xi = fab::paper::xi;        // 0.9483, limite da bend loss
static const double n_eff = fab::paper::n_eff;  // 2.25
static const double n_g = fab::paper::n_g;      // 4.05, Lumerical

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
        const auto &res = sim.run(in, false, false); // silente

        std::cout << std::left << std::setw(10) << std::fixed
                  << std::setprecision(2) << order << std::setw(12)
                  << casc.num_stages() << std::setw(14) << std::fixed
                  << std::setprecision(2) << casc.stage_order() << std::setw(16)
                  << std::fixed << std::setprecision(3)
                  << casc.usable_band() / 1e9 << std::setw(10) << std::fixed
                  << std::setprecision(2) << res.error_Dn * 100.0 << " %\n";
    }
    std::cout << "============================================================="
                 "==\n";
    std::printf("Riferimento [LIU25] per n = 0.54: %.2f %% teorico, "
                "%.2f %% FDTD\n\n",
                fab::paper::D_054_theory * 100.0,
                fab::paper::D_054_fdtd * 100.0);

    // =========================================================================
    // 2. SIMULAZIONE NOMINALE DETTAGLIATA (Caso n = 1.44)
    // =========================================================================
    const double target_n = 0.54; // Due anelli uniformi da n_i = 0.72
    std::cout << ">>> Configurazione simulazione nominale per n = " << target_n
              << " ...\n";

    MRRCascade cascade(target_n, R_ring, xi, n_eff, n_g);
    Simulation::Input pulse = Simulation::Input::gaussian_matched(cascade);

    Simulation nominal_sim(cascade);
    nominal_sim.print_setup(pulse);

    // align = false: i grafici mostrano le forme d'onda dove cadono
    // davvero. D_n e' comunque misurato sulla coppia allineata.
    const auto &nominal_res = nominal_sim.run(pulse, false, false);
    std::cout << "Risultato nominale: " << nominal_res.caption << "\n\n";

    // =========================================================================
    // 3. ANALISI STATISTICA MONTE CARLO SUL DISPOSITIVO PROGETTATO
    // =========================================================================
    std::cout << ">>> Avvio analisi di tolleranza Monte Carlo..." << std::endl;

    MonteCarlo::Config mc_cfg;
    mc_cfg.trials = 500;
    mc_cfg.yield_threshold = fab::paper::D_accept; // il 10% di [LIU25]
    mc_cfg.align_waveforms = false;
    mc_cfg.enforce_under_coupled = true; // Eq. (2) vale solo per r > xi

    // Errore geometrico estratto una volta per chip; r, xi, n_eff e n_g ne
    // discendono. Sigma e sensibilita' vengono da Fabrication.hpp.
    mc_cfg.correlated = true;

    // Heater per anello che riaggancia la risonanza al laser
    mc_cfg.enable_thermal_tuning = true;
    mc_cfg.sigma_df_tuned = fab::sigma_df_tuned;

    std::printf("Tolleranze (Fabrication.hpp): sigma_w = %.3f nm [LU17], "
                "sigma_h = %.3f nm [LU17]\n"
                "  -> sigma_r = %.2e, sigma_neff = %.2e, "
                "rho(anello-anello) = %.4f\n",
                fab::process::sigma_width * 1e9,
                fab::process::sigma_height * 1e9, fab::sigma_r(),
                fab::sigma_neff(), fab::layout::rho());

    MonteCarlo mc(cascade, pulse, mc_cfg);
    MonteCarlo::Result mc_res =
        mc.run(32768); // punti FFT: potenza di 2, per la radix-2 di Eigen

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
