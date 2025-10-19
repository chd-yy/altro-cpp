// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>

#include "altro/common/functionbase.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace problem {

// clang-format off
/**
 * @brief 表示如下形式的连续动力学函数：
 * \f[ \dot{x} = f(x, u) \f]
 *
 * 作为 `FunctionsBase` 接口的特化，用户需要实现以下接口：
 *
 * # 接口
 * - `int StateDimension() const` - 状态数量（x 的长度）
 * - `int ControlDimension() const` - 控制数量（u 的长度）
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<Eigen::VectorXd>
 * out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - 可选
 * - `bool HasHessian() const` - 指定是否实现 Hessian
 *
 * 我们使用以下 Eigen 类型别名：
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 用户还可以选择定义以下静态常量：
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * 这些可用于提供编译期大小信息。为获得最佳性能，强烈建议用户提供这些常量；
 * 若未提供，则默认使用 `Eigen::Dynamic`。
 *
 * ## FunctionBase API
 * 若需要原始的 FunctionBase API，需要在派生类的 public 接口中添加：
 *    using FunctionBase::Evaluate;
 *    using FunctionBase::Jacobian;
 *    using FunctionBase::Hessian;
 *
 * 注意：若在时变动力学中使用 FunctionBase API，务必在调用
 * `FunctionBase::Evaluate` 前，通过 `ContinuousDynamics::SetTime` 更新时间。
 */
// clang-format on
class ContinuousDynamics : public FunctionBase {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  int OutputDimension() const override { return StateDimension(); }

  // 新接口
  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t,
                        Eigen::Ref<VectorXd> xdot) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) = 0;

  // 便捷方法
  VectorXd Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t);
  VectorXd operator()(const VectorXdRef& x, const VectorXdRef& u, float t);

  // FunctionBase API
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    Evaluate(x, u, GetTime(), out);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    Jacobian(x, u, GetTime(), jac);
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {  // NOLINT(performance-unnecessary-value-param)
    Hessian(x, u, GetTime(), b, hess);
  }

  float GetTime() const { return t_; }
  void SetTime(float t) { t_ = t; }

 protected:
  float t_ = 0.0F;
};

  // clang-format off
/**
 * @brief 表示如下形式的离散动力学函数：
 * \f$ x_{k+1} = f(x_k, u_k) \f$
 * 
 * 这是 altro 库期望的动力学形式。连续时间动力学可通过 `DiscretizedDynamics` 等转换为离散模型。
 *
 * 作为 `FunctionsBase` 接口的特化，用户需要实现以下接口：
 *
 * # 接口
 * - `int StateDimension() const` - 状态数量（x 的长度）
 * - `int ControlDimension() const` - 控制数量（u 的长度）
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<Eigen::VectorXd>
 * out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - 可选
 * - `bool HasHessian() const` - 指定是否实现 Hessian
 *
 * 其中，`t` 为时间（针对时变动力学），`h` 为时间步。
 * 可通过 `SetTime`、`SetStep`、`GetTime`、`GetStep` 进行设置与获取。
 * 
 * 我们使用以下 Eigen 类型别名：
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 用户还可以选择定义以下静态常量：
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * 这些可用于提供编译期大小信息。为获得最佳性能，强烈建议用户提供这些常量；
 * 若未提供，则默认使用 `Eigen::Dynamic`。
 *
 * ## FunctionBase API
 * 若需要原始的 FunctionBase API，需要在派生类的 public 接口中添加：
 * 
 *    using FunctionBase::Evaluate;
 *    using FunctionBase::Jacobian;
 *    using FunctionBase::Hessian;
 *
 * 注意：若在时变动力学中使用 FunctionBase API，务必在调用
 * `FunctionBase::Evaluate` 前，通过 `DiscreteDynamics::SetTime` 与 
 * `DiscreteDynamics::SetStep` 更新时间与步长。
 */
// clang-format on
class DiscreteDynamics : public FunctionBase {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  int OutputDimension() const override { return StateDimension(); }

  // 新接口
  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                        Eigen::Ref<VectorXd> xdot) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) = 0;

  // 便捷方法
  VectorXd Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h);
  VectorXd operator()(const VectorXdRef& x, const VectorXdRef& u, float t, float h);

  // FunctionBase API
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    Evaluate(x, u, GetTime(), GetStep(), out);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    Jacobian(x, u, GetTime(), GetStep(), jac);
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {  // NOLINT(performance-unnecessary-value-param)
    Hessian(x, u, GetTime(), GetStep(), b, hess);
  }

  float GetTime() const { return t_; }
  void SetTime(float t) { t_ = t; }
  float GetStep() const { return h_; }
  void SetStep(float h) { h_ = h; }

 protected:
  float t_ = 0.0F;
  float h_ = 0.0F;
};

}  // namespace problem
}  // namespace altro