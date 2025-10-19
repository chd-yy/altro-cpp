// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <vector>
#include <array>
#include <memory>

#include "altro/eigentypes.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/common/state_control_sized.hpp"
namespace altro {
namespace problem {
/**
 * @brief 动力学系统显式积分方法的接口类。
 *
 * 所有子类必须实现 `Integrate` 方法，用于在某个时间步上对任意仿函数进行积分，
 * 以及通过 `Jacobian` 方法给出其一阶导数。
 * 
 * 子类应提供一个接收状态和控制维度的构造函数，例如：
 * 
 * `MyIntegrator(int n, int m);`
 *
 * @tparam DynamicsFunc 用于计算一阶常微分方程的类/可调用对象类型，其函数签名为：
 * dynamics(const VectorXd& x, const VectorXd& u, float t) const
 *
 * 期望接口参见 `ContinuousDynamics` 类。
 */
template <int NStates, int NControls>
class ExplicitIntegrator : public StateControlSized<NStates, NControls> {
 protected:
  using DynamicsPtr = std::shared_ptr<ContinuousDynamics>;

 public:
  ExplicitIntegrator(int n, int m) : StateControlSized<NStates, NControls>(n, m) {}
  ExplicitIntegrator() : StateControlSized<NStates, NControls>() {

  }
  virtual ~ExplicitIntegrator() = default;

  /**
   * @brief 在给定时间步上对动力学进行积分
   *
   * @param[in] dynamics 用于计算连续动力学的 `ContinuousDynamics` 对象
   * @param[in] x 状态向量
   * @param[in] u 控制向量
   * @param[in] t 自变量（例如时间）
   * @param[in] h 离散化步长（例如时间步）
   * @return VectorXd 时间步末的状态向量
   */
  virtual void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x,
                         const VectorXdRef& u, float t, float h,
                         Eigen::Ref<VectorXd> xnext) = 0;

  /**
   * @brief 计算离散动力学的雅可比矩阵
   *
   * 通常会调用连续动力学的雅可比计算。
   *
   * @pre 传入的 `jac` 必须已完成尺寸初始化
   *
   * @param[in] dynamics 用于计算连续动力学的 `ContinuousDynamics` 对象
   * @param[in] x 状态向量
   * @param[in] u 控制向量
   * @param[in] t 自变量（例如时间）
   * @param[in] h 离散化步长（例如时间步）
   * @param[out] jac 在 (x, u, t) 处评估得到的离散动力学雅可比
   */
  virtual void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x,
                        const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> jac) = 0;
};

/**
 * @brief 基本的显式欧拉积分器
 *
 * 最简单的积分器，只需要对连续动力学进行一次评估，但积分误差较大。
 *
 * @tparam DynamicsFunc
 */
class ExplicitEuler final : public ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic> {
 public:
  ExplicitEuler(int n, int m) : ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic>(n, m) {}
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {
    dynamics->Evaluate(x, u, t,  xnext);
    xnext = x + xnext * h;
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, Eigen::Ref<MatrixXd> jac) override {
    int n = x.size();
    int m = u.size();
    dynamics->Jacobian(x, u, t, jac);
    jac = MatrixXd::Identity(n, n + m) + jac * h;
  }
};

/**
 * @brief 四阶显式 Runge-Kutta 积分器（RK4）。
 *
 * 在众多机器人应用中是事实上的显式积分标准，兼顾精度与计算量。
 *
 * @tparam DynamicsFunc
 */
template <int NStates, int NControls>
class RungeKutta4 final : public ExplicitIntegrator<NStates, NControls> {
  using typename ExplicitIntegrator<NStates, NControls>::DynamicsPtr;
 public:

  RungeKutta4(int n, int m) : ExplicitIntegrator<NStates, NControls>(n, m) {
    Init();
  }
  RungeKutta4() : ExplicitIntegrator<NStates, NControls>() {
    Init();
  }
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k3_ * h, u, t + h, k4_);
    xnext = x + h * (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6;  // NOLINT(readability-magic-numbers)
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, Eigen::Ref<MatrixXd> jac) override {
    int n = dynamics->StateDimension();
    int m = dynamics->ControlDimension();

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)

    dynamics->Jacobian(x, u, t, jac);
    A_[0] = jac.topLeftCorner(n, n);
    B_[0] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k1_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[1] = jac.topLeftCorner(n, n);
    B_[1] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k2_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[2] = jac.topLeftCorner(n, n);
    B_[2] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + k3_ * h, u, t, jac);
    A_[3] = jac.topLeftCorner(n, n);
    B_[3] = jac.topRightCorner(n, m);

    dA_[0] = A_[0] * h;
    dA_[1] = A_[1] * (MatrixXd::Identity(n, n) + 0.5 * dA_[0]) * h;  // NOLINT(readability-magic-numbers)
    dA_[2] = A_[2] * (MatrixXd::Identity(n, n) + 0.5 * dA_[1]) * h;  // NOLINT(readability-magic-numbers)
    dA_[3] = A_[3] * (MatrixXd::Identity(n, n) + dA_[2]) * h;

    dB_[0] = B_[0] * h;
    dB_[1] = B_[1] * h + 0.5 * A_[1] * dB_[0] * h;  // NOLINT(readability-magic-numbers)
    dB_[2] = B_[2] * h + 0.5 * A_[2] * dB_[1] * h;  // NOLINT(readability-magic-numbers)
    dB_[3] = B_[3] * h + A_[3] * dB_[2] * h;

    jac.topLeftCorner(n, n) =
        MatrixXd::Identity(n, n)
        + (dA_[0] + 2 * dA_[1] + 2 * dA_[2] + dA_[3]) / 6;  // NOLINT(readability-magic-numbers)
    jac.topRightCorner(n, m) =
        (dB_[0] + 2 * dB_[1] + 2 * dB_[2] + dB_[3]) / 6;  // NOLINT(readability-magic-numbers)
  }

 private:
  void Init() {
    int n = this->StateDimension();
    int m = this->ControlDimension();
    k1_.setZero(n);
    k2_.setZero(n);
    k3_.setZero(n);
    k4_.setZero(n);
    for (int i = 0; i < 4; ++i) {
      A_[i].setZero(n, n); 
      B_[i].setZero(n, m);
      dA_[i].setZero(n, n); 
      dB_[i].setZero(n, m);
    }
  }

  // 这些成员需要可变，以便积分方法可保持 const
  // 它们替代了本应临时创建的数组，且不对外公开访问，这样处理是合理的。
  VectorNd<NStates> k1_;
  VectorNd<NStates> k2_;
  VectorNd<NStates> k3_;
  VectorNd<NStates> k4_;
  std::array<MatrixNxMd<NStates, NStates>, 4> A_;
  std::array<MatrixNxMd<NStates, NControls>, 4> B_;
  std::array<MatrixNxMd<NStates, NStates>, 4> dA_;
  std::array<MatrixNxMd<NStates, NControls>, 4> dB_;
};

}  // namespace problem
}  // namespace altro