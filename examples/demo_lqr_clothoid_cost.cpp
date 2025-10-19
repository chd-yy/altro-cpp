#include <iostream>
#include <cmath>

#include "examples/lqr_clothoid_cost.hpp"

int main() {
  using Cost = standalone::ClothoidLQRCost; // LQRCost<5,1>
  using DiagCost = standalone::LQRCostDiagonal<5, 1>;

  // Build Q and R for the screenshot costs:
  // - J = 1/2 w_omega * omega^2  → Q(4,4) = w_omega
  // - J = 1/2 w_alpha * alpha^2  → R(0,0) = w_alpha
  Cost::StateMatrix Q = Cost::StateMatrix::Zero();
  Cost::ControlMatrix R = Cost::ControlMatrix::Zero();

  double v = 10.0; // example speed [m/s]
  double c_omega = 1.0; // tuning constant for omega cost
  double c_alpha = 1.0; // tuning constant for alpha cost
  double w_omega = c_omega * std::pow(v, 6); // per screenshot note
  double w_alpha = c_alpha * std::pow(v, 8);

  Q(4, 4) = w_omega; // weight on omega^2
  R(0, 0) = w_alpha; // weight on alpha^2

  // Optional references (default zeros)
  Cost cost(Q, R);

  // Diagonal optimized variant construction (does identical math but faster for diagonal Q/R)
  DiagCost::StateVector QdiagVals = DiagCost::StateVector::Zero();
  DiagCost::ControlVector RdiagVals = DiagCost::ControlVector::Zero();
  QdiagVals(4) = w_omega; // same as Q(4,4)
  RdiagVals(0) = w_alpha; // same as R(0,0)
  DiagCost diagCost(QdiagVals, RdiagVals);

  // Example state x = [x, y, theta, kappa, omega]
  Cost::StateVector x; x << 0.0, 0.0, 0.0, 0.05, 0.02; // sample values
  // Example control u = [alpha]
  Cost::ControlVector u; u << 0.01;

  // Evaluate cost
  double J = cost.Evaluate(x, u);
  std::cout << "J = " << J << std::endl;

  // Evaluate diagonal optimized cost (should match)
  double Jd = diagCost.Evaluate(x, u);
  std::cout << "J (diag) = " << Jd << std::endl;

  // Gradients
  Cost::StateVector gx; Cost::ControlVector gu;
  cost.Gradient(x, u, gx, gu);
  std::cout << "grad x = " << gx.transpose() << std::endl;
  std::cout << "grad u = " << gu.transpose() << std::endl;

  // Hessians
  Cost::StateMatrix Hxx; standalone::LQRCost<5,1>::CrossMatrix Hxu; Cost::ControlMatrix Huu;
  cost.Hessian(x, u, Hxx, Hxu, Huu);
  std::cout << "Hxx(4,4) = " << Hxx(4,4) << ", Huu(0,0) = " << Huu(0,0) << std::endl;

  // Diagonal Hessian blocks
  DiagCost::StateMatrix Hxxd; DiagCost::CrossMatrix Hxud; DiagCost::ControlMatrix Huud;
  diagCost.Hessian(x, u, Hxxd, Hxud, Huud);
  std::cout << "Hxxd(4,4) = " << Hxxd(4,4) << ", Huud(0,0) = " << Huud(0,0) << std::endl;

  return 0;
}


