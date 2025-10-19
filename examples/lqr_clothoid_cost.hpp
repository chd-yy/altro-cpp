#pragma once

// Standalone LQR-style quadratic cost (templated dimensions).
// J = 1/2 (x - xref)^T Q (x - xref) + 1/2 (u - uref)^T R (u - uref)

#include <Eigen/Dense>
#include <stdexcept>

namespace standalone {

template <int kStateDim, int kControlDim>
class LQRCost {
public:
  using StateVector = Eigen::Matrix<double, kStateDim, 1>;
  using ControlVector = Eigen::Matrix<double, kControlDim, 1>;
  using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
  using ControlMatrix = Eigen::Matrix<double, kControlDim, kControlDim>;
  using CrossMatrix = Eigen::Matrix<double, kStateDim, kControlDim>;

  // Construct from explicit Q, R and references.
  LQRCost(const StateMatrix& Q, const ControlMatrix& R,
          const StateVector& xref = StateVector::Zero(),
          const ControlVector& uref = ControlVector::Zero())
      : Q_(Q), R_(R), xref_(xref), uref_(uref) {
    Validate();
  }

  // Evaluate cost.
  double Evaluate(const StateVector& x, const ControlVector& u) const {
    const StateVector dx = x - xref_;
    const ControlVector du = u - uref_;
    return 0.5 * dx.dot(Q_ * dx) + 0.5 * du.dot(R_ * du);
  }

  // Gradients w.r.t. x and u.
  void Gradient(const StateVector& x, const ControlVector& u,
                StateVector& dx, ControlVector& du) const {
    dx = Q_ * (x - xref_);
    du = R_ * (u - uref_);
  }

  // Hessian blocks.
  void Hessian(const StateVector& /*x*/, const ControlVector& /*u*/,
               StateMatrix& dxdx, CrossMatrix& dxdu, ControlMatrix& dudu) const {
    dxdx = Q_;
    dudu = R_;
    dxdu.setZero();
  }

  // Accessors
  const StateMatrix& Q() const { return Q_; }
  const ControlMatrix& R() const { return R_; }
  const StateVector& xref() const { return xref_; }
  const ControlVector& uref() const { return uref_; }

private:
  void Validate() const {
    if (!Q_.isApprox(Q_.transpose())) {
      throw std::invalid_argument("Q must be symmetric");
    }
    if (!R_.isApprox(R_.transpose())) {
      throw std::invalid_argument("R must be symmetric");
    }
  }

  StateMatrix Q_;
  ControlMatrix R_;
  StateVector xref_;
  ControlVector uref_;
};

// Backward compatible alias for the clothoid case (n=5, m=1)
using ClothoidLQRCost = LQRCost<5, 1>;

// Diagonal-optimized variant for Q and R. Keeps the same external behavior
// but stores Q, R as Eigen::DiagonalMatrix to reduce multiplications to
// elementwise ops when Q/R are diagonal (common in practice).
template <int kStateDim, int kControlDim>
class LQRCostDiagonal {
public:
  using StateVector = Eigen::Matrix<double, kStateDim, 1>;
  using ControlVector = Eigen::Matrix<double, kControlDim, 1>;
  using StateMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;
  using ControlMatrix = Eigen::Matrix<double, kControlDim, kControlDim>;
  using CrossMatrix = Eigen::Matrix<double, kStateDim, kControlDim>;
  using StateDiag = Eigen::DiagonalMatrix<double, kStateDim>;
  using ControlDiag = Eigen::DiagonalMatrix<double, kControlDim>;

  // Construct from diagonal matrices
  LQRCostDiagonal(const StateDiag& Qdiag, const ControlDiag& Rdiag,
                  const StateVector& xref = StateVector::Zero(),
                  const ControlVector& uref = ControlVector::Zero())
      : Qdiag_(Qdiag), Rdiag_(Rdiag), xref_(xref), uref_(uref) {}

  // Construct from diagonal vectors (values on the diagonal)
  LQRCostDiagonal(const StateVector& Qdiag_values, const ControlVector& Rdiag_values,
                  const StateVector& xref = StateVector::Zero(),
                  const ControlVector& uref = ControlVector::Zero())
      : Qdiag_(Qdiag_values.asDiagonal()),
        Rdiag_(Rdiag_values.asDiagonal()),
        xref_(xref),
        uref_(uref) {}

  double Evaluate(const StateVector& x, const ControlVector& u) const {
    const StateVector dx = x - xref_;
    const ControlVector du = u - uref_;
    // For diagonal D: x^T D x == (D.diagonal().cwiseProduct(x)).dot(x)
    const double x_term = Qdiag_.diagonal().cwiseProduct(dx).dot(dx);
    const double u_term = Rdiag_.diagonal().cwiseProduct(du).dot(du);
    return 0.5 * (x_term + u_term);
  }

  void Gradient(const StateVector& x, const ControlVector& u,
                StateVector& dx, ControlVector& du) const {
    dx = Qdiag_.diagonal().cwiseProduct(x - xref_);
    du = Rdiag_.diagonal().cwiseProduct(u - uref_);
  }

  void Hessian(const StateVector& /*x*/, const ControlVector& /*u*/,
               StateMatrix& dxdx, CrossMatrix& dxdu, ControlMatrix& dudu) const {
    dxdx.setZero();
    dudu.setZero();
    dxdu.setZero();
    dxdx.diagonal() = Qdiag_.diagonal();
    dudu.diagonal() = Rdiag_.diagonal();
  }

  const StateDiag& Qdiag() const { return Qdiag_; }
  const ControlDiag& Rdiag() const { return Rdiag_; }
  const StateVector& xref() const { return xref_; }
  const ControlVector& uref() const { return uref_; }

private:
  StateDiag Qdiag_;
  ControlDiag Rdiag_;
  StateVector xref_;
  ControlVector uref_;
};

} // namespace standalone


