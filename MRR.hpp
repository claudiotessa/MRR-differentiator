#ifndef MRR_HPP
#define MRR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdexcept>

class MRR {

    static constexpr double c = 2.99792458e8; // lightspeed

  public:

    MRR(double R,
        double t,
        double xi,
        double n_eff,
        double n_g = -1.0,
        double detuning = 0.0
        ): R(R), t(t), xi(xi), n_eff(n_eff), n_g(n_g > 0.0 ? n_g : n_eff),
          detuning(detuning), L_r(2.0 * M_PI * R) { tau((n_g * L_r) / c) }

    /**
     * @brief Creates a first order differentiator in critical coupling.
     * @param `B` the desired 3db bandwidth [Hz]. FWHD
     * @param `R` radius of the MRR [m].
     * @param `n_eff` effective refractive index.
     */
    static MRR first_order(double B, double R, double n_eff);

    /**
     * @brief Creates a fractional order differentiator in critical coupling.
     * @param `n` the order of the differentiator 0 < n < 1
     * @param `B` the desired 3db bandwidth [Hz].
     * @param `R` radius of the MRR [m].
     * @param `n_eff` effective refractive index.
     */
    static MRR fractional_order(double n, double R, double xi, double n_eff);

  private:
    double R;   // Radius
    double L_r; // Length (circumference)
    double t;   // slef coupling
    double xi;  // ring loss
    double n_eff; // effective index of the waveguide mode
    double n_g;   // group index of the waveguide mode
    double detuning;
    double tau; // round trip time
};

#endif
