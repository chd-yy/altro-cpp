// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <eigen3/Eigen/Dense>

#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief 将状态维度与控制维度相加
 * 如果任一为动态大小（Eigen::Dynamic），它们的和也将是动态大小。
 * 适用于模板元编程。
 * 
 * 提供用于检查运行时（实际）大小的 `StateDimension()` 与 `ControlDimension()`，
 * 以及用于检查编译期大小的 `StateMemorySize()` 与 `ControlMemorySize()`。
 *
 * @param n 编译期的状态维度
 * @param m 编译期的控制维度
 * @return 若两者在编译期均已知则返回 n+m，否则返回 Eigen::Dynamic
 */
constexpr int AddSizes(int n, int m) {
  if (n == Eigen::Dynamic || m == Eigen::Dynamic) {
    return Eigen::Dynamic;
  }
  return n + m;
}

/**
 * @brief 存储状态维度与控制维度的基类
 * 总结： 适用于需要固定维度（编译期已知）或在运行时注入维度且需一致性检查的场景。
 * 对于静态大小的数据结构，可选择在编译期存储这些维度。
 * 
 * 基本的继承示例：
 * @code {.cpp}
 * template <int n, int m>
 * class Derived : public StateControlSized<n,m> {
 *   public:
 *   Derived(int state_dim, int control_dim)
 *       : StateControlSized<n,m>(state_dim, control_dim) {}
 * }
 * @endcode
 * 
 * @param n 编译期的状态维度
 * @param m 编译期的控制维度
 */
template <int n, int m>
class StateControlSized {
 public:
  StateControlSized(int state_dim, int control_dim)
      : n_(state_dim), m_(control_dim) {
    if (n > 0) {
      ALTRO_ASSERT(n == n_, "State sizes must be consistent.");
    }
    if (m > 0) {
      ALTRO_ASSERT(m == m_, "Control sizes must be consistent.");
    }
  }
  StateControlSized() : n_(n), m_(m) {
    ALTRO_ASSERT(n > 0, "State dimension must be greater than zero.");
    ALTRO_ASSERT(m > 0, "Control dimension must be greater than zero.");
  }

  int StateDimension() const { return n_; }
  int ControlDimension() const { return m_; }
  static constexpr int StateMemorySize() { return n; }
  static constexpr int ControlMemorySize() { return m; }

 protected:
  int n_;
  int m_;
};

}  // namespace altro