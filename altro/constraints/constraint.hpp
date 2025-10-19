// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "altro/common/functionbase.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace constraints {

// Forward-declare for use in ZeroCone
class IdentityCone;

/**
 * @brief 等式约束（ZeroCone 的别名）
 *
 * 形式为 \f[ g(x,u) = 0 \f] 的通用等式约束
 *
 * 这种形式的等式约束的投影操作将值投影到零。对偶锥是恒等映射。
 */
class ZeroCone {
 public:
  ZeroCone() = delete;
  using DualCone = IdentityCone;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    ALTRO_UNUSED(x);
    x_proj.setZero();
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_UNUSED(x);
    jac.setZero();
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};

/**
 * @brief `ZeroCone` 锥的别名。
 * 
 */
using Equality = ZeroCone;

/**
 * @brief 恒等投影
 *
 * 恒等投影将点投影到自身。它是等式约束的对偶锥，用于锥形增广拉格朗日
 * 来处理等式约束。
 *
 */
class IdentityCone {
 public:
  IdentityCone() = delete;
  using DualCone = ZeroCone;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    x_proj = x;
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_ASSERT(jac.rows() == jac.cols(), "Jacobian must be square.");
    ALTRO_UNUSED(x);
    jac.setIdentity();
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};

/**
 * @brief 所有负数的空间，不等式约束的别名。
 *
 * 用于表示形式为 \f[ h(x) \leq 0 \f] 的不等式约束：
 *
 * 负象限是自对偶锥，其投影算子是逐元素的 `min(0, x)`。
 *
 */
class NegativeOrthant {
 public:
  NegativeOrthant() = delete;
  using DualCone = NegativeOrthant;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    for (int i = 0; i < x.size(); ++i) {
      x_proj(i) = std::min(0.0, x(i));
    }
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_ASSERT(jac.rows() == jac.cols(), "Jacobian must be square.");
    for (int i = 0; i < x.size(); ++i) {
      jac(i, i) = x(i) > 0 ? 0 : 1;
    }
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};

/**
 * @brief `NegativeOrthant` 锥的别名。
 * 
 */
using Inequality = NegativeOrthant;

/**
 * @brief 包含单个约束的基本信息
 * 
 */
struct ConstraintInfo {
  std::string label;
  int index;
  VectorXd violation;
  std::string type;

  std::string ToString(int precision = 4) const;
};

std::ostream& operator<<(std::ostream& os, const ConstraintInfo& coninfo);

// clang-format off
/**
 * @brief 形式为 \f[ g(x, u) \in K \f] 的抽象约束：
 *
 * 其中 \f$ K \f$ 是由 `ConType` 类型参数指定的任意凸锥。
 * 此公式支持通用等式和不等式约束。
 *
 * # 接口
 * 定义约束时，用户应实现以下接口：
 * - `int OutputDimension() const` - 输出大小（约束长度）。
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> out)`
 * - `std::string GetLabel() const` - 约束的简要描述，用于打印。
 *
 * 我们使用以下 Eigen 类型别名：
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * 约束至少需要具有连续的一阶导数，这些导数必须由用户实现。
 * 不提供自动或近似微分方法，但可以使用有限差分方法通过 `CheckJacobian`
 * 验证雅可比矩阵。有关更多信息，请参阅 `FunctionBase` 的文档。
 *
 * @tparam ConType 约束类型（等式、不等式、锥形等）
 */
// clang-format on
template <class ConType>
class Constraint : public FunctionBase {
 public:
  using ConstraintType = ConType;

  // These aren't used right now, but they need to be defined.
  int StateDimension() const override {
    ALTRO_ASSERT(false, "StateDimension hasn't been defined for this constraint.");
    return -1;
  }
  int ControlDimension() const override {
    ALTRO_ASSERT(false, "ControlDimension hasn't been defined for this constraint.");
    return -1;
  }

  // TODO(bjackson) [SW-14476] add 2nd order terms when implementing DDP
  bool HasHessian() const override { return false; }

  virtual std::string GetLabel() const { return GetConstraintType(); }

  std::string GetConstraintType() const {
    if (std::is_same<ConType, Equality>::value) {
      return "Equality Constraint";
    } else if (std::is_same<ConType, Inequality>::value) {
      return "Inequality Constraint";
    } else {
      return "Undefined Constraint Type";
    }
  }
};

template <class ConType>
using ConstraintPtr = std::shared_ptr<Constraint<ConType>>;

}  // namespace constraints
}  // namespace altro