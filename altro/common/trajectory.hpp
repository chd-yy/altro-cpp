// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>
#include <vector>

#include "altro/common/knotpoint.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

/**
 * @brief 表示状态和控制轨迹
 *
 * 如果轨迹中状态或控制的数量不是常数，可以将关联的类型参数设置为 Eigen::Dynamic。
 *
 * @tparam n 状态向量大小。可以是 Eigen::Dynamic。
 * @tparam m 控制向量大小。可以是 Eigen::Dynamic。
 * @tparam T 状态和控制向量的浮点精度
 */
template <int n, int m, class T = double>
class Trajectory {
  using StateVector = VectorN<n, T>;
  using ControlVector = VectorN<m, T>;

 public:
  /**
   * @brief 构造大小为 N 的新轨迹对象
   *
   * @param N 轨迹中的段数。这意味着有 N+1 个状态向量和 N 个控制向量。
   */
  explicit Trajectory(int N) : traj_(N+1) {}
  explicit Trajectory(int _n, int _m, int N)
      : traj_(N + 1, KnotPoint<n, m, T>(_n, _m)) {}
  explicit Trajectory(std::vector<KnotPoint<n, m, T>> zs) : traj_(zs) {}

  /**
   * @brief 从状态、控制和时间构造新的轨迹对象
   *
   * @param X (N+1,) 状态向量
   * @param U (N,) 控制向量
   * @param times (N+1,) 时间向量
   */
  Trajectory(std::vector<VectorN<n, T>> X, std::vector<VectorN<m, T>> U,
             std::vector<float> times) {
    ALTRO_ASSERT(X.size() == U.size() + 1,
                 "Length of control vector must be one less than the length of "
                 "the state trajectory.");
    ALTRO_ASSERT(X.size() == times.size(),
                 "Length of times vector must be equal to the length of the "
                 "state trajectory.");
    int N = U.size();
    traj_.reserve(N + 1);
    for (int k = 0; k < N; ++k) {
      float h = times[k + 1] - times[k];
      traj_.emplace_back(X[k], U[k], times[k], h);
    }
    traj_.emplace_back(X[N], 0 * U[N-1], times[N], 0.0);
  }


  /***************************** 复制 **************************************/
  Trajectory(const Trajectory& Z) : traj_(Z.traj_) {}
  Trajectory& operator=(const Trajectory& Z) {
    traj_ = Z.traj_;
    return *this;
  }

  /***************************** 移动 ***************************************/
  Trajectory(Trajectory&& Z) noexcept : traj_(std::move(Z.traj_)) {}
  Trajectory& operator=(Trajectory&& Z) noexcept {
    traj_ = std::move(Z.traj_);
    return *this;
  }

  /*************************** 迭代 **************************************/
  using iterator = typename std::vector<KnotPoint<n,m>>::iterator ;
  using const_iterator = typename std::vector<KnotPoint<n,m>>::const_iterator;
  iterator begin() { return traj_.begin(); }
  const_iterator begin() const { return traj_.begin(); }
  iterator end() { return traj_.end(); }
  const_iterator end() const { return traj_.end(); }

  /*************************** 获取器 ****************************************/
  int NumSegments() const { return traj_.size() - 1; }
  StateVector& State(int k) { return traj_[k].State(); }
  ControlVector& Control(int k) { return traj_[k].Control(); }

  const KnotPoint<n, m, T>& GetKnotPoint(int k) const { return traj_[k]; }
  const StateVector& State(int k) const { return traj_[k].State(); }
  const ControlVector& Control(int k) const { return traj_[k].Control(); }

  KnotPoint<n, m, T>& GetKnotPoint(int k) { return traj_[k]; }
  KnotPoint<n, m, T>& operator[](int k) { return GetKnotPoint(k); }

  int StateDimension(int k) const { return traj_[k].StateDimension(); }
  int ControlDimension(int k) const { return traj_[k].ControlDimension(); }

  T GetTime(int k) const { return traj_[k].GetTime(); }
  float GetStep(int k) const { return traj_[k].GetStep(); }

  /*************************** 设置器 ****************************************/

  /**
   * @brief 将状态和控制设置为零
   * 
   */
  void SetZero() {
    for (iterator z_ptr = begin(); z_ptr != end(); ++z_ptr) {
      z_ptr->State().setZero();
      z_ptr->Control().setZero();
    }
  }

  void SetTime(int k, float t) { traj_[k].SetTime(t); }
  void SetStep(int k, float h) { traj_[k].SetStep(h); }

  void SetUniformStep(float h) {
    int N = NumSegments();
    for (int k = 0; k < N; ++k) {
      traj_[k].SetStep(h);
      traj_[k].SetTime(static_cast<float>(k) * h);
    }
    traj_[N].SetStep(0.0);
    traj_[N].SetTime(static_cast<float>(h) * N);
  }

  /**
   * @brief 检查时间和时间步长是否一致
   *
   * @param eps 浮点比较的容差检查
   * @return 如果对所有 k 都有 t[k+1] - t[k] == h[k] 则返回 true
   */
  bool CheckTimeConsistency(const double eps = 1e-6,
                            const bool verbose = false) {
    for (int k = 0; k < NumSegments(); ++k) {
      float h_calc = GetTime(k + 1) - GetTime(k);
      float h_stored = GetStep(k);
      if (std::abs(h_stored - h_calc) > eps) {
        if (verbose) {
          std::cout << "k=" << k << "\t h=" << h_stored << std::endl;
          std::cout << "t-=" << GetTime(k) << "\t t+=" << GetTime(k + 1)
                    << "\t dt=" << h_calc << std::endl;
        }
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<KnotPoint<n, m, T>> traj_;
};

using TrajectoryXXd = Trajectory<Eigen::Dynamic, Eigen::Dynamic, double>;

}  // namespace altro