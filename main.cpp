#include "MRR.hpp"
#include "Simulation.hpp"

// Waveguide parameters shared by every figure.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // effective / group index

int main() {
    const double n = 0.54; // fractional order
    const double B = 10e9; // target 3 dB bandwidth [Hz]

    // --- Main result: the differentiated waveform ---
    MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff);

    // The paper drives its 0.54-order device with a Gaussian sized against the
    // ring's Eq. (4) width. Other shapes are available - super_gaussian(),
    // sech(), rectangular() - but the Gaussian is the one the paper's error
    // figures are measured with, so it is the only fair comparison.
    Simulation sim(ring, n);

    // Create the impulse coupled to the ring
    Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);

    sim.propagate(pulse, false);
    sim.plot_waveforms();

    sim.fractional_response(1.421e9);
    sim.show();

    return 0;
}
