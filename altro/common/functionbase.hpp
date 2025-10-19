// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <type_traits>

#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

namespace altro {

// clang-format off
/**
 * @brief 表示形式为 \f[ out = f(x, u) \f] 的通用向量值函数
 *
 * 至少，函数必须具有明确定义的雅可比矩阵。通过定义雅可比转置向量积的雅可比矩阵
 * 来提供二阶信息。
 *
 * 可以使用有限差分通过 `CheckJacobian` 和 `CheckHessian` 检查实现的导数。
 * 这些函数可以提供样本输入，否则如果没有提供则生成随机输入。
 *
 * # 接口
 * 要实现此接口，用户必须指定以下内容：
 * - `int StateDimension() const` - 状态数量（x 的长度）
 * - `int ControlDimension() const` - 控制数量（u 的长度）
 * - `int OutputDimension() const` - 输出大小（out 的长度）。
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - 可选
 * - `bool HasHessian() const` - 指定是否实现了 Hessian
 *
 * 我们使用以下 Eigen 类型别名：
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 用户还可以选择定义静态常量：
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * 可用于提供编译时大小信息。这些值可以使用 `StateMemorySize`、`ControlMemorySize`
 * 和 `OutputMemorySize` 函数在运行时类型上查询。
 */
// clang-format off
class FunctionBase {
 public:
  virtual ~FunctionBase() = default;

  static constexpr int NStates = Eigen::Dynamic;
  static constexpr int NControls = Eigen::Dynamic;
  static constexpr int NOutputs = Eigen::Dynamic;

  virtual int StateDimension() const { return 0; }
  virtual int ControlDimension() const { return 0; }
  virtual int OutputDimension() const = 0;

  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) {  // NOLINT(performance-unnecessary-value-param)
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }
  virtual bool HasHessian() const = 0;

  bool TestCheck(const VectorXdRef& x, const VectorXdRef& u);

  bool CheckJacobian(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckJacobian(const VectorXdRef& x, const VectorXdRef& u, double eps = kDefaultTolerance,
                     bool verbose = false);
  bool CheckHessian(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckHessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
                    double eps = kDefaultTolerance, bool verbose = false);

 protected:
  static constexpr double kDefaultTolerance = 1e-4;
};

// clang-format on
/**
 * @brief 表示抽象标量值函数
 *
 * `FunctionBase` 接口对标量值函数的特化。
 *
 * 为了符号方便，我们将梯度定义为标量函数一阶导数的列向量。这是相应雅可比矩阵的转置。
 *
 * Hessian 就是梯度的雅可比矩阵。
 *
 * `CheckGradient` 和 `CheckHessian` 函数可用于验证用户实现的导数。注意，当向
 * `CheckHessian` 传递参数时，仍需要指定 `b` 向量参数，即使在标量函数接口中不需要。
 *
 * # 接口
 * 用户必须定义以下函数：
 * - `int StateDimension() const` - 状态数量（x 的长度）
 * - `int ControlDimension() const` - 控制数量（u 的长度）
 * - `double Evaluate(const VectorXdRef& x, const VectorXdRef& u)`
 * - `void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::MatrixXd> hess)`
 * - `bool HasHessian() const` - 指定是否实现了 Hessian - 可选（假设为 true）
 *
 * 我们使用以下 Eigen 类型别名：
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 用户还可以选择定义静态常量：
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *
 * 可用于提供编译时大小信息。这些值可以使用 `StateMemorySize` 和 `ControlMemorySize`
 * 函数在运行时类型上查询。
 *
 */
// clang-format off
class ScalarFunction : public FunctionBase {
 public:
  static const int NOutputs = 1;
  int OutputDimension() const override { return 1; }

  // New Interface
  virtual double Evaluate(const VectorXdRef& x, const VectorXdRef& u) = 0;
  virtual void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> grad) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> hess) = 0;

  // Pass parent interface to new interface
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    ALTRO_ASSERT(out.size() == 1, "Output must be of size 1 for scalar functions");
    out(0) = Evaluate(x, u);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_ASSERT(jac.rows() == 1, "Jacobian of a scalar function must have a single row.");
    // Reinterpret the 1xN Jacobian as an Nx1 column vector
    Eigen::Map<VectorNd<jac.ColsAtCompileTime>> grad(jac.data(), jac.cols());
    Gradient(x, u, grad);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {
    ALTRO_ASSERT(b.size() == 1 && b.isApproxToConstant(1),
                 "The b vector for scalar Hessians must be a vector of a single 1.");
    ALTRO_UNUSED(b);
    Hessian(x, u, hess);
  }
  bool HasHessian() const override { return true; }

  // Derivative checking
  bool CheckGradient(double eps = kDefaultTolerance, bool verbose = false);
  bool CheckGradient(const VectorXdRef& x, const VectorXdRef& u, double eps = kDefaultTolerance,
                     bool verbose = false);
};

}  // namespace altro