#include "MRR.hpp"
#include "Simulation.hpp"

// Waveguide parameters shared by every figure.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // effective / group index

int main() {
    const double n = 0.54;  // fractional order
    const double B = 10e9;  // target 3 dB bandwidth [Hz]

    // --- Main result: the differentiated waveform ---
    // Pass `false` as the last argument to keep the waveforms unshifted, with
    // the ring's real latency visible.
    MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff);
    Simulation::Propagation pulse = Simulation::propagate(ring, n, true);

    Simulation::input_signal(pulse);
    Simulation::waveforms(pulse);
    Simulation::optical_power(pulse);

    // --- Supporting: |H| and phase vs the ideal ---
    MRR mrr_1 = MRR::first_order(B, R_ring, n_eff);
    Simulation::first_order_response(mrr_1, B);

    MRR mrr_frac = MRR::fractional_order(n, R_ring, 0.90, n_eff);
    Simulation::fractional_response(mrr_frac, n, B);

    Simulation::show();
    return 0;
}
