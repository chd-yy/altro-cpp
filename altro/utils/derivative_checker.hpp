// 版权 [2021] Optimus Ride Inc.

#pragma once

#include "altro/eigentypes.hpp"

namespace altro {
namespace utils {

template <int nrows, int ncols, class Func>
Eigen::Matrix<double, nrows, ncols> FiniteDiffJacobian(
    const Func &f, const Eigen::Ref<const Eigen::Matrix<double, ncols, 1>> &x,
    const double eps = 1e-6, const bool central = false) {
  const int n = x.rows();

  // 计算函数值并获得输出维度
  Eigen::Matrix<double, nrows, 1> y = f(x);
  const int m = y.rows();
  Eigen::Matrix<double, nrows, ncols> jac =
      Eigen::Matrix<double, nrows, ncols>::Zero(m, n);

  // 创建扰动向量
  Eigen::Matrix<double, ncols, 1> e = Eigen::Matrix<double, ncols, 1>::Zero(n);

  // 按列循环
  e(0) = eps;
  for (int i = 0; i < n; ++i) {
    double step = eps;
    if (central) {
      y = f(x - e);
      step = 2 * eps;
    }
    jac.col(i) = (f(x + e) - y) / step;
    if (i < n - 1) {
      e(i + 1) = e(i);
      e(i) = 0;
    }
  }
  return jac;
}

/**
 * @brief 使用有限差分计算函数 f 的近似雅可比矩阵
 *
 * @tparam ncols 输入的静态尺寸。对堆分配向量可设为 -1。
 * @tparam Func 函数样对象类型（只需实现 () 运算符）。
 * @tparam nrows 输出的静态尺寸。可为 -1（默认），表示堆分配数组。
 * @tparam T 浮点类型。为获得最佳效果，应使用双精度。
 * @param f 接受向量 x 并返回向量 y 的函数样对象（实现 () 运算符）。
 * @param x 传入到函数 `func` 的输入向量。
 * @param eps 有限差分步长的扰动大小。
 * @param central 若为 `true`，使用更精确但计算量更大的中心差分方法。
 * @return Eigen::Matrix<T, nrows, ncols>
 */
template <class Func>
Eigen::MatrixXd FiniteDiffJacobian(const Func &f,
                                   const VectorXdRef &x,
                                   const double eps = 1e-6,
                                   const bool central = false) {
  return FiniteDiffJacobian<-1, -1, Func>(f, x, eps, central);
}

/**
 * @brief 将返回标量的函数对象转换为返回一维向量的函数对象
 *
 * @tparam Func 函数样对象
 */
template <class Func>
struct ScalarToVec {
  using Vector1d = Eigen::Matrix<double, 1, 1>;
  Vector1d operator()(const VectorXd &x) const {
    Vector1d y;
    y << f(x);
    return y;
  }
  Func f;
};

/**
 * @brief 使用有限差分计算标量值函数的梯度
 * 简单地将函数转换为返回一维向量并调用 `FiniteDiffJacobian`
 *
 * 详见 `FiniteDiffJacobian` 的完整文档。
 *
 * @return Eigen::Matrix<T, ncols, 1> 列向量
 */
template <int ncols, class Func>
Eigen::Matrix<double, ncols, 1> FiniteDiffGradient(
    const Func &f, const Eigen::Matrix<double, ncols, 1> &x,
    const double eps = 1e-6, const bool central = false) {
  ScalarToVec<Func> f2 = {f};
  return FiniteDiffJacobian<1, ncols, ScalarToVec<Func>>(f2, x, eps, central)
      .transpose();
}

/**
 * @brief 计算任意标量值函数梯度的仿函数
 *
 * @tparam nrows 输入的静态尺寸。可为 Eigen::Dynamic。
 * @tparam Func 函数样对象
 * @tparam T 浮点精度类型
 */
template <int nrows, class Func, class T>
struct FiniteDiffGradientFunc {
  using GradVec = Eigen::Matrix<T, nrows, 1>;
  GradVec operator()(const GradVec &x) const {
    return FiniteDiffGradient(f, x, eps, central);
  }
  Func f;
  double eps;
  bool central;
};

/**
 * @brief 计算标量值函数 f 的海森矩阵
 * 生成一个使用有限差分计算梯度的仿函数，然后对该对象调用 `FiniteDiffJacobian`。
 *
 * 详见 `FiniteDiffJacobian` 的完整文档。
 *
 * @return Eigen::Matrix<T, ncols, ncols>
 */
template <int ncols, class Func>
Eigen::Matrix<double, ncols, ncols> FiniteDiffHessian(
    const Func &f, const Eigen::Matrix<double, ncols, 1> &x,
    const double eps = 1e-4, const bool central = true) {
  using GradFunc = FiniteDiffGradientFunc<ncols, Func, double>;
  GradFunc gradfun = {f, eps, central};
  return FiniteDiffJacobian<ncols, ncols, GradFunc>(gradfun, x, eps, central);
}

}  // namespace utils
}  // namespace altro