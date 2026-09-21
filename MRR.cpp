#include <Eigen/Dense>
#include <iostream>

#include "MRR.hpp"

void stampa_matrice() {
  Eigen::Matrix2d matrix;

  matrix << 1, 2, 3, 4;

  std::cout << matrix << '\n';
}

int main() {
  stampa_matrice();
  return 0;
}
