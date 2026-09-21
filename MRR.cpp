#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <complex>
#include <unsupported/Eigen/FFT>

#include "MRR.hpp"

using namespace Eigen;
using namespace std::complex_literals;

template <typename Derived>
auto fftshift(const Eigen::DenseBase<Derived>& vec) {
  using Scalar = typename Derived::Scalar;
  Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 
    Derived::ColsAtCompileTime> out(vec.rows(), vec.cols());

  Eigen::Index n = vec.size();
  Eigen::Index mid = (n + 1) / 2; // matches MATLAB's ceil(n/2) split

  // Swap second half to first half, and first half to second half
  out.head(n - mid) = vec.tail(n - mid);
  out.tail(mid)     = vec.head(mid);

  return out;
}

int main(){

  long N = 1e5;   // number of samples
  Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N, -10e-9, 10e-9);

  double dt = time(1) - time(0);    // number of samples
  Eigen::ArrayXcd Df = Eigen::ArrayXd::LinSpaced(N, -1/(2*dt), 1/(2*dt)).cast<std::complex<double>>();

  double c = 3e8;
  double bandwidth = 10e9;

  double E_in = 0;  // input signal

  double n_eff = 2.4;
  double R = 1e-4;
  double L_ring = 2.0 * M_PI*R;

  // explain here
  double tau = L_ring / ( c / n_eff);
  double tau_c = 1.0 / (M_PI * bandwidth);
  double tau_n = tau_c / tau;

  // coupling coefficient
  double r = sqrt(tau_n / (1.0 + tau_n) );
  double t = sqrt(1 - r * r);

  // give defs
  ArrayXcd beta = 2.0 * M_PI * Df * n_eff / c;
  double phi = 0.0; 
  double gamma = r;  // alpha + j * beta
  
  ArrayXcd phase = (-1.0i * beta * L_ring + ArrayXcd::Ones(beta.size()) * 1.0i * phi);
  ArrayXcd H_through = (r - gamma * phase.exp() ) / (1.0 - r * gamma * phase.exp() );
  ArrayXcd H_diff = 1.0i * r / (1.0 - r*r) * tau *
    (2.0 * M_PI * Df) * (-1.0i * tau * M_PI * Df).exp();



  Eigen::FFT<double> fft;
  Eigen::VectorXcd fft_out;
  fft.fwd(fft_out, E_in);
  Eigen::VectorXcd E_in_fft = fftshift(fft_out);

  

  return 0;

}

