// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>

#include "altro/eigentypes.hpp"
#include "altro/common/functionbase.hpp"
#include "altro/utils/derivative_checker.hpp"

namespace altro {
namespace problem {

/**
 * @brief 表示一个标量值代价函数。
 *
 * 作为 ScalarFunction 接口的特化，用户需要实现下面描述的接口。
 * 与 `ScalarFunction` 接口的主要区别是，关于状态和控制的偏导数
 * 是分别传递的，而不是作为单个参数传递。原始 API 只是传递联合
 * 导数的适当部分。
 *
 * # 接口
 * 用户必须定义以下函数：
 * - `int StateDimension() const` - 状态数量（x的长度）
 * - `int ControlDimension() const` - 控制数量（u的长度）
 * - `double Evaluate(const VectorXdRef& x, const VectorXdRef& u)`
 * - `void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> dx,
 * Eigen::Ref<Eigen::VectorXd> du)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::MatrixXd> dxdx,
 * Eigen::Ref<Eigen::MatrixXd> dxdu, Eigen::Ref<Eigen::MatrixXd> dudu)`
 * - `bool HasHessian() const` - 指定是否实现了Hessian - 可选（假设为true）
 * 
 * 我们使用以下Eigen类型别名：
 *    using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 用户还可以选择定义静态常量：
 *    static constexpr int NStates
 *    static constexpr int NControls
 *
 * 这些可用于提供编译时大小信息。为了获得最佳性能，
 * 强烈建议用户为其实现指定这些常量。
 * 
 * # ScalarFunction API
 * 要使用 ScalarFunction API，请在派生类的公共接口中插入以下行：
 *    using ScalarFunction::Gradient;
 *    using ScalarFunction::Hessian;
 */
class CostFunction : public altro::ScalarFunction {
 public:
  using altro::ScalarFunction::Hessian;

  // 新接口
  virtual void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                        Eigen::Ref<VectorXd> du) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                       Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) = 0;

  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> grad) override {
    Gradient(x, u, grad.head(StateDimension()), grad.tail(ControlDimension()));
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> hess) override {
    const int n = StateDimension();
    const int m = ControlDimension();
    constexpr int Nx = NStates;
    constexpr int Nu = NControls;
    Hessian(x, u, hess.topLeftCorner<Nx, Nx>(n, n), hess.topRightCorner<Nx, Nu>(n, m),
            hess.bottomRightCorner<Nu, Nu>(m, m));
  }
};
}  // namespace problem
}  // namespace altro