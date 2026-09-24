#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Fabrication.hpp"
#include "MRRCascade.hpp"
#include "MonteCarlo.hpp"
#include "Plotter.hpp"
#include "Simulation.hpp"
#include <optional>

// Reference device: the theory-guided design of [LIU25] Sec. 2, via
// Fabrication.hpp. The tolerances were derived on this geometry, so changing
// it invalidates them.
static const double R_ring = fab::paper::R;    // 1.9 um
static const double xi = fab::paper::xi;       // 0.9428, set by bend loss
static const double n_eff = fab::paper::n_eff; // 2.25
static const double n_g = fab::paper::n_g;     // 4.05, Lumerical

struct CliArgs {
    double n = -1.0;
    double t0_ps = -1.0;
    bool custom_t0 = false;
};

void print_usage(const char *prog_name) {
    std::cout << "Uso: " << prog_name << " -n <ordine> [-t <T0_in_ps>] [-h]\n\n"
              << "Opzioni:\n"
              << "  -n, --order <valore>    Ordine di derivazione "
                 "(obbligatorio, n > 0)\n"
              << "  -t, -T0, --pulse <ps>   Durata T0 impulso gaussiano in ps "
                 "(opzionale)\n"
              << "  -h, --help              Mostra questa guida\n\n"
              << "Esempi:\n"
              << "  " << prog_name << " -n 1.44\n"
              << "  " << prog_name << " -n 0.54 -t 3.0\n";
}

std::optional<CliArgs> parse_flags(int argc, char *argv[]) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return std::nullopt;
        } else if (arg == "-n" || arg == "--order") {
            if (i + 1 < argc) {
                try {
                    args.n = std::stod(argv[++i]);
                } catch (const std::exception &) {
                    std::cerr << "Errore: valore non valido per -n ('"
                              << argv[i] << "')\n";
                    return std::nullopt;
                }
            } else {
                std::cerr << "Errore: flag " << arg
                          << " senza valore associato.\n";
                return std::nullopt;
            }
        } else if (arg == "-t" || arg == "-T0" || arg == "--pulse") {
            if (i + 1 < argc) {
                try {
                    args.t0_ps = std::stod(argv[++i]);
                    args.custom_t0 = true;
                } catch (const std::exception &) {
                    std::cerr << "Errore: valore non valido per -t ('"
                              << argv[i] << "')\n";
                    return std::nullopt;
                }
            } else {
                std::cerr << "Errore: flag " << arg
                          << " richiede un valore in ps.\n";
                return std::nullopt;
            }
        } else {
            std::cerr << "Attenzione: opzione non riconosciuta '" << arg
                      << "'\n";
        }
    }

    if (args.n <= 0.0) {
        std::cerr << "Errore: ordine n mancante o non positivo.\n\n";
        print_usage(argv[0]);
        return std::nullopt;
    }

    return args;
}

int main(int argc, char *argv[]) {
    auto args = parse_flags(argc, argv);
    if (!args) {
        return 1;
    }

    std::cout << "=========================================================\n";
    std::cout << " SIMULAZIONE DIFFERENZIATORE OTTICO FRAZIONARIO (IDEALE) \n";
    std::cout << "=========================================================\n";

    MRRCascade cascade(args->n, R_ring, xi, n_eff, n_g);

    Simulation::Input in;
    if (args->custom_t0) {
        in = Simulation::Input::gaussian(args->t0_ps * 1e-12);
    } else {
        if (std::abs(args->n - 0.54) < 1e-3) {
            in = Simulation::Input::gaussian(fab::paper::input_T0); // 3 ps
        } else if (std::abs(args->n - 1.44) < 1e-3 ||
                   std::abs(args->n - 2.10) < 1e-3) {
            in = Simulation::Input::gaussian(fab::paper::input_T0_144); // 7 ps
        } else {
            in = Simulation::Input::gaussian_matched(cascade);
        }
    }

    std::cout << "\n--- Dispositivo Ottico (MRRCascade) ---\n"
              << "Ordine totale (n)        : " << std::fixed
              << std::setprecision(4) << cascade.order() << "\n"
              << "Numero stadi (N)         : " << cascade.num_stages() << "\n"
              << "Ordine per stadio (n_i)  : " << cascade.stage_order() << "\n"
              << "Autocoppiamento (r)      : " << cascade.stage_self_coupling()
              << "\n"
              << "Perdite giro (xi)        : " << cascade.round_trip_loss()
              << "\n"
              << "Banda utile aggregata    : " << cascade.usable_band() / 1e9
              << " GHz\n";

    std::cout << "\n--- Segnale d'Ingresso ---\n"
              << "Impulso                  : " << in.describe() << "\n";

    // Simulazione e verifica errore Dn
    Simulation sim(cascade);
    const auto &res = sim.run(in, true, false);

    std::cout << "\n--- Risultati di Propagazione ---\n"
              << "Ritardo di gruppo (lag)  : " << std::fixed
              << std::setprecision(2) << res.lag_ps << " ps\n"
              << "Errore di derivata (Dn)  : \033[1;32m" << res.error_Dn * 100.0
              << " %\033[0m\n";

    //  Visualizzazione
    Plotter::plot_all(sim, true);

    std::cout << "=========================================================\n";
    std::cout << " SIMULAZIONE MONTE CARLO \n";
    std::cout << "=========================================================\n";

    // === MONTE CARLO ===
    const double target_n = args->n;
    std::cout << ">>> Nominal simulation for n = " << std::setprecision(2)
              << target_n << " ...\n";

    // The paper's own input for this device: 3 ps for a single ring (Sec. 2),
    // 7 ps once it is a cascade (Sec. 3.B). Keyed off the order so that
    // retargeting target_n does not silently leave the wrong pulse behind.
    Simulation::Input pulse = Simulation::Input::gaussian(
        target_n <= 1.0  ? fab::paper::input_T0
        : target_n < 2.0 ? fab::paper::input_T0_144
                         : fab::paper::input_T0_210);

    Simulation nominal_sim(cascade);
    nominal_sim.print_setup(pulse);

    // align = false: the plots show the waveforms where they really fall.
    // D_n is measured on the aligned pair regardless.
    const auto &nominal_res = nominal_sim.run(pulse, false, false);
    std::cout << "Nominal result: " << nominal_res.caption << "\n\n";

    // =========================================================================
    // MONTE CARLO TOLERANCE ANALYSIS OF THE DESIGNED DEVICE
    // =========================================================================
    std::cout << ">>> Starting Monte Carlo tolerance analysis..." << std::endl;

    MonteCarlo::Config mc_cfg;
    mc_cfg.trials = 500;
    mc_cfg.yield_threshold = fab::paper::D_accept; // the 10% bar of [LIU25]
    mc_cfg.align_waveforms = false;

    // One geometry error per chip; r, xi, n_eff and n_g all follow from it.
    // Sigmas and sensitivities come from Fabrication.hpp.
    mc_cfg.correlated = true;

    // A per-ring heater re-locks the resonance to the laser.
    mc_cfg.enable_thermal_tuning = true;
    mc_cfg.sigma_df_tuned = fab::sigma_df_tuned;

    std::printf("Tolerances (Fabrication.hpp): sigma_w = %.3f nm [LU17], "
                "sigma_h = %.3f nm [LU17]\n"
                "  -> sigma_r = %.2e, sigma_neff = %.2e, "
                "rho(ring-ring) = %.4f\n",
                fab::process::sigma_width * 1e9,
                fab::process::sigma_height * 1e9, fab::sigma_r(),
                fab::sigma_neff(), fab::layout::rho());

    MonteCarlo mc(cascade, pulse, mc_cfg);
    // FFT length: a power of two, for Eigen's radix-2 path.
    MonteCarlo::Result mc_res = mc.run(32768);

    mc_res.print_summary();

    // =========================================================================
    // FIGURES
    // =========================================================================
    std::cout << ">>> Opening figures (close the windows to finish)..."
              << std::endl;
    // Plotter::plot_all(nominal_sim, false); // waveforms and dB/phase spectrum
    Plotter::plot_monte_carlo(mc_res, mc_cfg.yield_threshold * 100.0);
    Plotter::show();

    return 0;
}
