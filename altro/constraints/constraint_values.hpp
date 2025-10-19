// Copyright [2021] Optimus Ride Inc.

#pragma once
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/common/state_control_sized.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/eigentypes.hpp"

namespace altro {
namespace constraints {

/**
 * @brief 一个约束，还为约束值、雅可比矩阵、对偶变量等分配内存。
 *
 * 此类还提供评估增广拉格朗日等项的方法。
 *
 * @tparam n 编译时状态维度
 * @tparam m 编译时控制维度
 * @tparam ConType 约束类型（等式、不等式、锥形等）
 */
template <int n, int m, class ConType>
class ConstraintValues : public Constraint<ConType> {
  static constexpr int p = Eigen::Dynamic;
  static constexpr int n_m = AddSizes(n, m);

 public:
  static constexpr double kDefaultPenaltyScaling = 10.0;
  /**
   * @brief 构造新的约束值对象
   *
   * @param state_dim 状态维度
   * @param control_dim  控制维度
   * @param con 指向约束的指针。假设约束函数可以用与 state_dim 和 control_dim 一致的输入进行评估
   */
  ConstraintValues(const int state_dim, const int control_dim, ConstraintPtr<ConType> con)
      : n_(state_dim), m_(control_dim), con_(std::move(con)) {
    int output_dim = con_->OutputDimension();
    c_.setZero(output_dim);
    lambda_.setZero(output_dim);
    penalty_.setOnes(output_dim);
    jac_.setZero(output_dim, state_dim + control_dim);
    hess_.setZero(state_dim + control_dim, state_dim + control_dim);
    lambda_proj_.setZero(output_dim);
    c_proj_.setZero(output_dim);
    proj_jac_.setZero(output_dim, output_dim);
    jac_proj_.setZero(output_dim, state_dim + control_dim);
  }

  /***************************** 获取器 **************************************/
  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }

  ConstraintPtr<ConType> GetConstraint() { return con_; }
  VectorNd<p>& GetDuals() { return lambda_; }
  VectorNd<p>& GetPenalty() { return penalty_; }
  VectorNd<p>& GetConstraintValue() { return c_; }
	double GetPenaltyScaling() const { return penalty_scaling_; }

  VectorNd<p>& GetViolation() {
    ConType::Projection(c_, c_proj_);
    c_proj_ = c_ - c_proj_;
    return c_proj_;
  }

  ConstraintInfo GetConstraintInfo() {
    return ConstraintInfo{con_->GetLabel(), 0, GetViolation(), con_->GetConstraintType()};
  }

  /***************************** 设置器 **************************************/
  /**
   * @brief 为所有约束设置相同的惩罚
   *
   * @param rho 惩罚值。rho >= 0。
   */
  void SetPenalty(double rho) {
    ALTRO_ASSERT(rho >= 0, "Penalty must be positive.");
    penalty_.setConstant(rho);
  }

  void SetPenaltyScaling(double phi) { 
    ALTRO_ASSERT(phi >= 1, "Penalty must be greater than 1.");
    penalty_scaling_ = phi; 
  }

  /***************************** 方法 **************************************/
  /**
   * @brief 评估增广拉格朗日
   *
   * 形式为以下优化问题的增广拉格朗日：
   * \f{aligned}{
   *   \text{minimize} &&& f(x) \\
   *   \text{subject to} &&& c(x) \in K \\
   * \f}
   *
   * 定义为
   * \f[
   * f(x) + \frac{1}{2 \rho} (||\Pi_{K^*}(\lambda - \rho c(x))||_2^2 - ||\lambda||_2^2)
   * \f]
   * 其中 \f$ \lambda \f$ 是拉格朗日乘数（对偶变量），\f$ \rho \f$ 是标量惩罚参数，
   * \f$ \Pi_{K^*} \f$ 是对偶锥 \f$ K^* \f$ 的投影算子。
   *
   * @param x 状态向量
   * @param u 控制向量
   * @return 当前节点在 x 和 u 处评估的增广拉格朗日。
   */
  double AugLag(const VectorXdRef& x, const VectorXdRef& u) {
    const double rho = penalty_(0);
    con_->Evaluate(x, u, c_);

    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    double J = lambda_proj_.squaredNorm() - lambda_.squaredNorm();
    J = J / (2 * rho);
    return J;
  }

  /**
   * @brief 增广拉格朗日的梯度
   *
   * 使用对偶锥投影算子的雅可比矩阵。
   *
   * @param[in] x 状态向量
   * @param[in] u 控制向量
   * @param[out] dx 关于状态的梯度。
   * @param[out] du 关于控制的梯度。
   */
  void AugLagGradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                      Eigen::Ref<VectorXd> du) {
    const double rho = penalty_(0);

    // TODO(bjackson): Avoid these redundant calls.
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    ConType::DualCone::Jacobian(lambda_ - rho * c_, proj_jac_);
    const int output_dim = con_->OutputDimension();
    dx = -(proj_jac_ * jac_.topLeftCorner(output_dim, this->n_)).transpose() * lambda_proj_;
    du = -(proj_jac_ * jac_.topRightCorner(output_dim, this->m_)).transpose() * lambda_proj_;
  }
  /**
   * @brief 增广拉格朗日的 Hessian 矩阵
   *
   * 使用对偶锥和约束的投影算子的雅可比转置向量积的雅可比矩阵。
   *
   * @param[in] x 状态向量
   * @param[in] u 控制向量
   * @param[out] dxdx 关于状态的 Hessian 矩阵。
   * @param[out] dxdu 关于状态和控制的 Hessian 交叉项。
   * @param[out] dudu 关于控制的 Hessian 矩阵。
   */
  void AugLagHessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                     Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu, const bool full_newton) {
    const double rho = penalty_(0);

    // TODO(bjackson): Avoid these redundant calls.
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    ConType::DualCone::Jacobian(lambda_ - rho * c_, proj_jac_);
    jac_proj_ = proj_jac_ * jac_;
    const int output_dim = con_->OutputDimension();
    dxdx = rho * jac_proj_.topLeftCorner(output_dim, this->n_).transpose()
           * jac_proj_.topLeftCorner(output_dim, this->n_);
    dxdu = rho * jac_proj_.topLeftCorner(output_dim, this->n_).transpose()
           * jac_proj_.topRightCorner(output_dim, this->m_);
    dudu = rho * jac_proj_.topRightCorner(output_dim, this->m_).transpose()
           * jac_proj_.topRightCorner(output_dim, this->m_);

    if (full_newton) {
      throw std::runtime_error("Second-order constraint terms are not yet supported.");
    }
  }

  /**
   * @brief 更新对偶变量
   * 
   * 使用当前约束和惩罚值更新对偶变量。
   * 结果对偶变量被投影回对偶锥，使得它们总是保证相对于对偶锥可行。
   * 
   * 更新形式为：
   * \f[
   * \lambda^+ - \Pi_{K^*}(\lambda - \rho c)
   * \f]
   * 
   */
  void UpdateDuals() {
    ConType::DualCone::Projection(lambda_ - penalty_.asDiagonal() * c_, lambda_);
  }

  /**
   * @brief 更新惩罚参数
   * 
   * 目前只是进行简单的均匀几何增长。
   * 
   */
  void UpdatePenalties() {
    // TODO(bjackson): Look into more advanced methods for updating the penalty parameter
    penalty_ *= penalty_scaling_;
    const double rho = penalty_(0);
    (void) rho;
  }

  /**
   * @brief 计算最大约束违反
   * 
   * @tparam p 计算违反时使用的范数（默认 = Infinity）
   * @return 最大约束违反
   */
  template <int p = Eigen::Infinity>
  double MaxViolation() {
    ConType::Projection(c_, c_proj_);
    c_proj_ = c_ - c_proj_;
    return c_proj_.template lpNorm<p>();
  }

  /**
   * @brief 查找最大惩罚
   * 
   * @return 最大惩罚参数
   */
  double MaxPenalty() { return penalty_.maxCoeff(); }

  // Pass constraint interface to internal pointer
  static constexpr int NStates = n;
  static constexpr int NControls = m;
  int OutputDimension() const override { return con_->OutputDimension(); }
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    con_->Evaluate(x, u, c);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<MatrixXd> jac) override {
    con_->Jacobian(x, u, jac);
  }

  /**
   * @brief 评估约束及其导数
   *
   * 在内部存储结果。
   *
   * @param[in] x 状态向量
   * @param[in] u 控制向量
   */
  void CalcExpansion(const VectorXdRef& x, const VectorXdRef& u) {
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
  }

  void ResetDualVariables() {
    lambda_.setZero();
  }

 private:
  const int n_;  // 状态维度
  const int m_;  // 控制维度
  ConstraintPtr<ConType> con_;
  VectorNd<p> c_;            // 约束值
  VectorNd<p> lambda_;       // 拉格朗日乘数
  VectorNd<p> penalty_;      // 惩罚值
  MatrixNxMd<p, n_m> jac_;    // 雅可比矩阵
  MatrixNxMd<n_m, n_m> hess_;  // Hessian 矩阵

  VectorNd<p> lambda_proj_;     // 投影乘数
  VectorNd<p> c_proj_;          // 投影约束值
  MatrixNxMd<p, p> proj_jac_;   // 投影操作的雅可比矩阵
  MatrixNxMd<p, n_m> jac_proj_;  // 通过投影操作的雅可比矩阵 (jac_ * proj_jac_)

  double penalty_scaling_ = kDefaultPenaltyScaling;
};

}  // namespace constraints
}  // namespace altro