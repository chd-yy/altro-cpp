// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <memory>

#include "altro/common/knotpoint.hpp"
#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace ilqr {

/**
 * @brief 存储代价函数的一阶和二阶展开
 * 底层内存可以存储在堆上或栈上，
 * 取决于状态和控制维度是否在编译时通过类型参数指定。
 * 使用 Eigen::Dynamic 可在堆上分配。
 *
 * @tparam n 编译时状态维度
 * @tparam m 编译时控制维度
 */
template <int n, int m>
class CostExpansion : public StateControlSized<n, m> {
 public:
  explicit CostExpansion(int state_dim, int control_dim)
      : StateControlSized<n, m>(state_dim, control_dim),
        dxdx_(Eigen::Matrix<double, n, n>::Zero(state_dim, state_dim)),
        dxdu_(Eigen::Matrix<double, n, m>::Zero(state_dim, control_dim)),
        dudu_(Eigen::Matrix<double, m, m>::Zero(control_dim, control_dim)),
        dx_(Eigen::Matrix<double, n, 1>::Zero(state_dim, 1)),
        du_(Eigen::Matrix<double, m, 1>::Zero(control_dim, 1)) {}

  // 拷贝操作符
  CostExpansion(const CostExpansion& exp)
      : StateControlSized<n, m>(exp.StateDimension(), exp.ControlDimension()),
        dxdx_(exp.dxdx()),
        dxdu_(exp.dxdu()),
        dudu_(exp.dudu()),
        dx_(exp.dx()),
        du_(exp.du()) {}
  CostExpansion& operator=(const CostExpansion& exp) {
    if (n > 0) {
      ALTRO_ASSERT(n == exp.StateDimension(),
                   "Invalid copy. State dimension must be consistent.");
    }
    if (m > 0) {
      ALTRO_ASSERT(m == exp.ControlDimension(),
                   "Invalid copy. Control dimension must be consistent.");
    }
    this->n_ = exp.StateDimension();
    this->m_ = exp.ControlDimension();
    this->dxdx_ = exp.dxdx();
    this->dxdu_ = exp.dxdu();
    this->dudu_ = exp.dudu();
    this->dx_ = exp.dx();
    this->du_ = exp.du();
    return *this;
  }

  // 移动操作符
  CostExpansion(CostExpansion&& exp) noexcept
      : StateControlSized<n, m>(exp.StateDimension(), exp.ControlDimension()),
        dxdx_(std::move(exp.dxdx_)),
        dxdu_(std::move(exp.dxdu_)),
        dudu_(std::move(exp.dudu_)),
        dx_(std::move(exp.dx_)),
        du_(std::move(exp.du_)) {}
  CostExpansion& operator=(CostExpansion&& exp) noexcept {
    if (n > 0) {
      ALTRO_ASSERT(n == exp.StateDimension(),
                   "Invalid copy. State dimension must be consistent.");
    }
    if (m > 0) {
      ALTRO_ASSERT(m == exp.ControlDimension(),
                   "Invalid copy. Control dimension must be consistent.");
    }
    this->n_ = exp.StateDimension();
    this->m_ = exp.ControlDimension();
    this->dxdx_ = std::move(exp.dxdx());
    this->dxdu_ = std::move(exp.dxdu());
    this->dudu_ = std::move(exp.dudu());
    this->dx_ = std::move(exp.dx());
    this->du_ = std::move(exp.du());
    return *this;
  }


  Eigen::Matrix<double, n, n>& dxdx() { return dxdx_; }
  Eigen::Matrix<double, n, m>& dxdu() { return dxdu_; }
  Eigen::Matrix<double, m, m>& dudu() { return dudu_; }
  Eigen::Matrix<double, n, 1>& dx() { return dx_; }
  Eigen::Matrix<double, m, 1>& du() { return du_; }
  const Eigen::Matrix<double, n, n>& dxdx() const { return dxdx_; }
  const Eigen::Matrix<double, n, m>& dxdu() const { return dxdu_; }
  const Eigen::Matrix<double, m, m>& dudu() const { return dudu_; }
  const Eigen::Matrix<double, n, 1>& dx() const { return dx_; }
  const Eigen::Matrix<double, m, 1>& du() const { return du_; }

  /**
   * @brief 计算代价函数的梯度和 Hessian 矩阵并存储在当前展开中
   *
   * @tparam n2 状态向量的编译时大小
   * @tparam m2 控制向量的编译时大小
   * @param[in] costfun 要计算展开的代价函数
   * @param[in] z 评估展开的状态和控制
   */
  template <int n2, int m2>
  void CalcExpansion(const std::shared_ptr<problem::CostFunction>& costfun,
                     const KnotPoint<n2, m2>& z) {
    CalcExpansion(costfun, z.State(), z.Control());
  }

  void CalcExpansion(const std::shared_ptr<problem::CostFunction>& costfun,
                     const VectorXdRef& x,
                     const VectorXdRef& u) {
    ALTRO_ASSERT(x.rows() == this->n_, "Inconsistent state dimension.");
    ALTRO_ASSERT(u.rows() == this->m_, "Inconsistent control dimension.");
    costfun->Gradient(x, u, dx_, du_);
    costfun->Hessian(x, u, dxdx_, dxdu_, dudu_);
  }

  void SetZero() {
    dxdx_.setZero();
    dxdu_.setZero();
    dudu_.setZero();
    dx_.setZero();
    du_.setZero();
  }

 private:
  Eigen::Matrix<double, n, n> dxdx_;
  Eigen::Matrix<double, n, m> dxdu_;
  Eigen::Matrix<double, m, m> dudu_;
  Eigen::Matrix<double, n, 1> dx_;
  Eigen::Matrix<double, m, 1> du_;
};

}  // namespace ilqr
}  // namespace altro