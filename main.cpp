#include "MRR.hpp"
#include "Simulation.hpp"

// Waveguide parameters shared by every figure.
static const double R_ring = 100e-6; // ring radius [m]
static const double n_eff = 2.4;     // effective / group index

int main() {
  const double n = 0.54;  // fractional order
  const double B = 10e9;  // target 3 dB bandwidth [Hz]

  // --- Main result: the differentiated waveform ---
  MRR ring = MRR::fractional_order(n, R_ring, 0.99, n_eff);

  // The paper drives its 0.54-order device with a Gaussian sized against the
  // ring's Eq. (4) width. Other shapes are available - super_gaussian(),
  // sech(), rectangular() - but the Gaussian is the one the paper's error
  // figures are measured with, so it is the only fair comparison.
  // Simulation::Input pulse = Simulation::Input::gaussian_matched(ring);

  // Pass `false` as the `align` argument to keep the waveforms unshifted,
  // with the ring's real latency visible.
  Simulation::Propagation out = Simulation::propagate(ring, pulse, n, false);

  // Simulation::input_signal(out);
  Simulation::waveforms(out);
  // Simulation::optical_power(out);

  // --- Supporting: |H| and phase vs the ideal ---
  // MRR mrr_1 = MRR::first_order(B, R_ring, n_eff);
  // Simulation::first_order_response(mrr_1, B);

  MRR mrr_frac = MRR::fractional_order(n, R_ring, 0.90, n_eff);
  Simulation::fractional_response(mrr_frac, n, B);

  Simulation::show();
  return 0;
}
