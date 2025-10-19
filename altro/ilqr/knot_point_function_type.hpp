// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <memory>
#include <array>

#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/dynamics_expansion.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace ilqr {

// TODO(bjackson): 实现其他正则化方法
enum class BackwardPassRegularization {
  kControlOnly,
  // kStateOnly,
  // kStateControl
};

/**
 * @brief 存储在每个节点评估各种表达式的方法和数据
 *
 * 存储代价和动力学定义，并提供方法来评估它们的展开。
 * 还提供计算 iLQR 后向传递所需项的方法，并存储动作值展开、
 * 代价到达的二次近似以及反馈和前馈增益。
 *
 * @tparam n 编译时状态维度
 * @tparam m 编译时控制维度
 */
template <int n, int m>
class KnotPointFunctions : public StateControlSized<n, m> {
  using DynamicsPtr = std::shared_ptr<problem::DiscreteDynamics>;
  using CostFunPtr = std::shared_ptr<problem::CostFunction>;
  using JacType = Eigen::Matrix<double, n, AddSizes(n, m)>;

public:
  KnotPointFunctions(DynamicsPtr dynamics, CostFunPtr costfun)
      : StateControlSized<n, m>(
            // Use comma operator to check the dynamics pointer before using it.
            // Must apply to both arguments since argument evaluation order is undefined.
            (CheckDynamicsPtr(dynamics), dynamics->StateDimension()),
            (CheckDynamicsPtr(dynamics), dynamics->ControlDimension())),
        model_ptr_(std::move(dynamics)),
        costfun_ptr_(std::move(costfun)),
        cost_expansion_(model_ptr_->StateDimension(),
                        model_ptr_->ControlDimension()),
        dynamics_expansion_(model_ptr_->StateDimension(),
                            model_ptr_->ControlDimension()),
        action_value_expansion_(model_ptr_->StateDimension(),
                                model_ptr_->ControlDimension()),
        action_value_expansion_regularized_(model_ptr_->StateDimension(),
                                            model_ptr_->ControlDimension()) {
    ALTRO_ASSERT(costfun_ptr_ != nullptr, "Cannot provide a null cost function pointer.");
    Init();
  }
  // 为最后一个节点创建 kpf
  KnotPointFunctions(int state_dim, int control_dim, CostFunPtr costfun)
      : StateControlSized<n, m>(state_dim, control_dim), model_ptr_(nullptr),
        costfun_ptr_(std::move(costfun)), cost_expansion_(state_dim, control_dim),
        dynamics_expansion_(state_dim, control_dim),
        action_value_expansion_(state_dim, control_dim),
        action_value_expansion_regularized_(state_dim, control_dim) {
    ALTRO_ASSERT(costfun_ptr_ != nullptr, "Cannot provide a null cost function pointer.");
    Init();
  }

  /**
   * @brief 评估节点的代价
   *
   * @param x 状态向量
   * @param u 控制向量
   * @return double
   */
  double Cost(const VectorXdRef &x,
              const VectorXdRef &u) const {
    return costfun_ptr_->Evaluate(x, u);
  }

  /**
   * @brief 评估节点处的离散动力学
   *
   * @param x 状态向量
   * @param u 控制向量
   * @param t 自变量（例如时间）
   * @param h 自变量的步长
   * @param xnext 下一个节点的状态
   */
  void Dynamics(const VectorXdRef &x,
                const VectorXdRef &u, float t, float h,
                Eigen::Ref<VectorXd> xnext) const { // NOLINT(performance-unnecessary-value-param)
    model_ptr_->Evaluate(x, u, t, h, xnext);
  }

  /**
   * @brief 评估代价函数的二阶展开
   *
   * @param x 状态向量
   * @param u 控制向量
   */
  void CalcCostExpansion(const VectorXdRef &x,
                         const VectorXdRef &u) {
    cost_expansion_.SetZero();
    cost_expansion_.CalcExpansion(costfun_ptr_, x, u);
  }

  /**
   * @brief 评估动力学的一阶展开
   *
   * @param x 状态向量
   * @param u 控制向量
   * @param t 自变量（例如时间）
   * @param h 自变量的步长
   */
  void CalcDynamicsExpansion(const VectorXdRef &x,
                             const VectorXdRef &u, const float t,
                             const float h) {
    if (model_ptr_) {
      dynamics_expansion_.SetZero();
      dynamics_expansion_.CalcExpansion(model_ptr_, x, u, t, h);
    }
  }

  /**
   * @brief 计算终端代价到达，或最后一个节点的代价到达。
   *
   */
  void CalcTerminalCostToGo() {
    ctg_hessian_ = cost_expansion_.dxdx();
    ctg_gradient_ = cost_expansion_.dx();
  }

  /**
   * @brief 给定下一时刻代价到达的二次近似，计算动作值展开。
   *
   * @pre 必须先计算代价和动力学展开。
   *
   * @param ctg_hessian 下一时刻代价到达的 Hessian 矩阵。
   * @param ctg_gradient 下一时刻代价到达的梯度。
   */
  void
  CalcActionValueExpansion(const Eigen::Ref<const MatrixXd> &ctg_hessian,
                           const Eigen::Ref<const MatrixXd> &ctg_gradient) {
    Eigen::Block<JacType, n, n> A = dynamics_expansion_.GetA();
    Eigen::Block<JacType, n, m> B = dynamics_expansion_.GetB();
    action_value_expansion_.dxdx() =
        cost_expansion_.dxdx() + A.transpose() * ctg_hessian * A;
    action_value_expansion_.dxdu() =
        cost_expansion_.dxdu() + A.transpose() * ctg_hessian * B;
    action_value_expansion_.dudu() =
        cost_expansion_.dudu() + B.transpose() * ctg_hessian * B;
    action_value_expansion_.dx() =
        cost_expansion_.dx() + A.transpose() * ctg_gradient;
    action_value_expansion_.du() =
        cost_expansion_.du() + B.transpose() * ctg_gradient;
  }

  /**
   * @brief 在求解最优反馈策略之前，向动作值展开添加正则化。
   *
   * @pre 必须先计算动作值展开。
   *
   * @param rho 正则化量
   * @param reg_type 如何合并正则化。
   */
  void RegularizeActionValue(const double rho,
                             BackwardPassRegularization reg_type =
                                 BackwardPassRegularization::kControlOnly) {
    action_value_expansion_regularized_ = action_value_expansion_;
    switch (reg_type) {
    case BackwardPassRegularization::kControlOnly: {
      action_value_expansion_regularized_.dudu() +=
          Eigen::Matrix<double, m, m>::Identity(this->m_, this->m_) * rho;
      break;
    }
    }
  }

  /**
   * @brief 通过使用 Cholesky 分解反转动作值展开关于 u 的 Hessian 矩阵
   * 来计算反馈和前馈增益。
   *
   * @pre 必须先计算正则化的动作值展开。
   *
   * @return 描述 Cholesky 分解结果的 Eigen 枚举。
   */
  Eigen::ComputationInfo CalcGains() {
    // TODO(bjackson): 在类中存储分解
    Eigen::LLT<Eigen::Matrix<double, m, m>> Quu_chol;
    Quu_chol.compute(action_value_expansion_regularized_.dudu());
    Eigen::ComputationInfo info = Quu_chol.info();
    if (info == Eigen::Success) {
      feedback_gain_ = Quu_chol.solve(
          action_value_expansion_regularized_.dxdu().transpose());
      feedback_gain_ *= -1;
      feedforward_gain_ =
          Quu_chol.solve(action_value_expansion_regularized_.du());
      feedforward_gain_ *= -1;
    }
    return info;
  }

  /**
   * @brief 给定反馈策略，计算代价到达的当前二次近似。
   *
   * @pre 必须先计算增益和动作值展开。
   *
   */
  void CalcCostToGo() {
    Eigen::Matrix<double, m, n> &K = GetFeedbackGain();
    Eigen::Matrix<double, m, 1> &d = GetFeedforwardGain();
    CostExpansion<n, m> &Q = GetActionValueExpansion();
    ctg_gradient_ = Q.dx() + K.transpose() * Q.dudu() * d +
                    K.transpose() * Q.du() + Q.dxdu() * d;
    ctg_hessian_ = Q.dxdx() + K.transpose() * Q.dudu() * K +
                   K.transpose() * Q.dxdu().transpose() + Q.dxdu() * K;
    ctg_delta_[0] = d.dot(Q.du());
    ctg_delta_[1] = 0.5 * d.dot(Q.dudu() * d); // NOLINT(readability-magic-numbers)
  }

  void AddCostToGo(std::array<double, 2>* const deltaV) const {
    (*deltaV)[0] += ctg_delta_[0];
    (*deltaV)[1] += ctg_delta_[1];
  }

  void AddCostToGo(double* const deltaV) const {
    deltaV[0] += ctg_delta_[0];
    deltaV[1] += ctg_delta_[1];
  }

  /**************************** 获取器 ***************************************/
  std::shared_ptr<problem::DiscreteDynamics> GetModelPtr() {
    return model_ptr_;
  }
  std::shared_ptr<problem::CostFunction> GetCostFunPtr() {
    return costfun_ptr_;
  }
  CostExpansion<n, m> &GetCostExpansion() { return cost_expansion_; }
  DynamicsExpansion<n, m> &GetDynamicsExpansion() {
    return dynamics_expansion_;
  }

  Eigen::Matrix<double, n, n> &GetCostToGoHessian() { return ctg_hessian_; }
  Eigen::Matrix<double, n, 1> &GetCostToGoGradient() { return ctg_gradient_; }
  double GetCostToGoDelta(const double alpha = 1.0) {
    return alpha * ctg_delta_[0] + alpha * alpha * ctg_delta_[1];
  }
  CostExpansion<n, m> &GetActionValueExpansion() {
    return action_value_expansion_;
  }
  CostExpansion<n, m> &GetActionValueExpansionRegularized() {
    return action_value_expansion_regularized_;
  }
  Eigen::Matrix<double, m, n> &GetFeedbackGain() { return feedback_gain_; }
  Eigen::Matrix<double, m, 1> &GetFeedforwardGain() {
    return feedforward_gain_;
  }

private:
  void Init() {
    feedback_gain_ = Eigen::Matrix<double, m, n>::Zero(this->m_, this->n_);
    feedforward_gain_ = Eigen::Matrix<double, m, 1>::Zero(this->m_, 1);
    ctg_hessian_ = Eigen::Matrix<double, n, n>::Zero(this->n_, this->n_);
    ctg_gradient_ = Eigen::Matrix<double, n, 1>::Zero(this->n_, 1);
    ctg_delta_[0] = 0;
    ctg_delta_[1] = 0;
  }

  void CheckDynamicsPtr(const DynamicsPtr& dynamics) {
    (void) dynamics;  // 需要这样做以抑制错误的未使用变量警告
    ALTRO_ASSERT(dynamics != nullptr, "Cannot provide a null dynamics pointer.");
  }

  std::shared_ptr<problem::DiscreteDynamics> model_ptr_;
  std::shared_ptr<problem::CostFunction> costfun_ptr_;

  CostExpansion<n, m> cost_expansion_;
  DynamicsExpansion<n, m> dynamics_expansion_;
  CostExpansion<n, m> action_value_expansion_;
  CostExpansion<n, m> action_value_expansion_regularized_;

  Eigen::Matrix<double, m, n> feedback_gain_;
  Eigen::Matrix<double, m, 1> feedforward_gain_;

  Eigen::Matrix<double, n, n> ctg_hessian_;
  Eigen::Matrix<double, n, 1> ctg_gradient_;
  std::array<double, 2> ctg_delta_;
};

} // namespace ilqr
} // namespace altro