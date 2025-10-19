// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <string>

#include <fmt/format.h>

#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief 存储单个节点的状态、控制、时间和时间步长
 * 总结 ： 轨迹离散节点管理：在 iLQR/AL 等算法中存取每个时间节点的状态与控制，并区分终端节点。
 * 状态和控制向量可以存在于栈或堆上，利用 Eigen 的 Matrix 类。如果 @tparam n 或 @tparam m
 * 等于 Eigen::Dynamic，向量将在堆上分配。
 *
 * 使用 `StateDimension` 和 `ControlDimension` 查询实际状态或控制维度。
 * 使用 `StateMemorySize` 和 `ControlMemorySize` 获取类型参数。
 *
 * @tparam n 状态向量大小。可以是 Eigen::Dynamic。
 * @tparam m 控制向量大小。可以是 Eigen::Dynamic。
 * @tparam T 状态和控制变量的精度
 */
template <int n, int m, class T = double>
class KnotPoint : public StateControlSized<n, m> {
  using StateVector = VectorN<n, T>;
  using ControlVector = VectorN<m, T>;

 public:
  KnotPoint()
      : StateControlSized<n, m>(n, m),
        x_(StateVector::Zero()),
        u_(ControlVector::Zero()) {}
  KnotPoint(const StateVector& x, const ControlVector& u, const float t = 0.0,
            const float h = 0.0)
      : StateControlSized<n, m>(x.size(), u.size()),
        x_(x),
        u_(u),
        t_(t),
        h_(h) {}
  KnotPoint(int _n, int _m)
      : StateControlSized<n, m>(_n, _m),
        x_(StateVector::Zero(_n)),
        u_(ControlVector::Zero(_m)) {}

  // 从不同内存位置但相同大小的节点复制
  template <int n2, int m2>
  KnotPoint(const KnotPoint<n2, m2>& z2)  // NOLINT(google-explicit-constructor)
      : StateControlSized<n, m>(z2.StateDimension(), z2.ControlDimension()),
        x_(z2.State()),
        u_(z2.Control()),
        t_(z2.GetTime()),
        h_(z2.GetStep()) {}

  // 复制操作
  KnotPoint(const KnotPoint& z)
      : StateControlSized<n, m>(z.n_, z.m_),
        x_(z.x_),
        u_(z.u_),
        t_(z.t_),
        h_(z.h_) {}
  KnotPoint& operator=(const KnotPoint& z) {
    x_ = z.x_;
    u_ = z.u_;
    t_ = z.t_;
    h_ = z.h_;
    this->n_ = z.n_;
    this->m_ = z.m_;
    return *this;
  }

  // 移动操作
  KnotPoint(KnotPoint&& z) noexcept
      : StateControlSized<n, m>(z.n_, z.m_),
        x_(std::move(z.x_)),
        u_(std::move(z.u_)),
        t_(z.t_),
        h_(z.h_) {}
  KnotPoint& operator=(KnotPoint&& z) noexcept {
    x_ = std::move(z.x_);
    u_ = std::move(z.u_);
    t_ = z.t_;
    h_ = z.h_;
    this->n_ = z.n_;
    this->m_ = z.m_;
    return *this;
  }

  static KnotPoint Random() {
    ALTRO_ASSERT(n > 0 && m > 0,
                 "Must pass in size if state or control dimension is unknown "
                 "at compile time.");
    return Random(n, m);
  }

  static KnotPoint Random(int state_dim, int control_dim) {
    VectorN<n> x = VectorN<n>::Random(state_dim);
    VectorN<m> u = VectorN<m>::Random(control_dim);
    const double max_time = 10.0;
    const double max_h = 1.0;
    const int resolution = 100;
    double t = UniformRandom(max_time, resolution);
    double h = UniformRandom(max_h, resolution);
    return KnotPoint(x, u, t, h);
  }

  StateVector& State() { return x_; }
  ControlVector& Control() { return u_; }
  const StateVector& State() const { return x_; }
  const ControlVector& Control() const { return u_; }
  VectorN<AddSizes(n,m), T> GetStateControl() const {
    VectorN<AddSizes(n,m), T> z;
    z << x_, u_;
    return z;
  }
  float GetTime() const { return t_; }
  float GetStep() const { return h_; }
  void SetTime(float t) { t_ = t; }
  void SetStep(float h) { h_ = h; }

  /**
   * @brief 检查节点是否是轨迹中的最后一个点，它没有长度且只存储状态向量。
   *
   * @return 如果节点是终端节点则返回 true
   */
  bool IsTerminal() const { return h_ == 0; }

  /**
   * @brief 将节点设置为终端节点，或轨迹中的最后一个节点。
   *
   */
  void SetTerminal() {
    h_ = 0;
    u_.setZero();
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const KnotPoint<n, m, T>& z) {
    return os << z.ToString();
  }

  /**
   * @brief 创建包含单行中所有状态和控制打印输出的字符串。
   * 
   * @param width 控制每个数值字段的宽度
   * @return std::string 
   */
  std::string ToString(int width = 9) const {
    std::string out;
    out += fmt::format("x: [");
    for (int i = 0; i < this->n_; ++i) {
      out += fmt::format("{1: > {0}.3g} ", width, State()(i));
    }
    out += fmt::format("] u: [");
    for (int i = 0; i < this->m_; ++i) {
      out += fmt::format("{1: > {0}.3g} ", width, Control()(i));
    }
    out += fmt::format("] t={:4.2}, h={:4.2}", GetTime(), GetStep());
    return out;
  }

 private:
  static double UniformRandom(double upper, int resolution) { 
    return upper * static_cast<double>(rand() % resolution) / static_cast<double>(resolution); 
  }

  StateVector x_;
  ControlVector u_;
  float t_ = 0.0;  // 时间
  float h_ = 0.0;  // 时间步长
};

}  // namespace altro