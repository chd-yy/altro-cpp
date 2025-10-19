// Copyright [2021] Optimus Ride Inc.

#pragma once

#include "altro/augmented_lagrangian/al_cost.hpp"
#include "altro/problem/problem.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief 构建增广拉格朗日轨迹优化问题。
 *
 * 接受一个约束轨迹优化问题，通过使用增广拉格朗日代价将约束移动到代价函数中，
 * 创建一个无约束轨迹优化问题。每个代价函数都是一个 ALCost<n, m>。
 *
 * @tparam n 编译时状态维度。
 * @tparam m 编译时控制维度。
 * @param[in] prob 原始的、可能受约束的优化问题。
 * @param[out] costs 可选容器，将填充分配给问题的 ALCost 类型。
 * 很有用，因为问题只存储通用 CostFunction 指针。
 *
 * @return problem::Problem 一个新的无约束轨迹优化问题，具有包含原始问题约束的增广拉格朗日代价函数。
 */
template <int n, int m>
problem::Problem BuildAugLagProblem(const problem::Problem& prob,
                                    std::vector<std::shared_ptr<ALCost<n, m>>>* costs = nullptr) {
  const int N = prob.NumSegments();
  problem::Problem prob_al(N, prob.GetInitialStatePointer());

  // 复制初始状态和动力学
  prob_al.SetInitialState(prob.GetInitialState());
  for (int k = 0; k < N; ++k) {
    prob_al.SetDynamics(prob.GetDynamics(k), k);
  }

  // 创建结合原始代价函数和约束的增广拉格朗日代价函数
  for (int k = 0; k <= N; ++k) {
    std::shared_ptr<ALCost<n, m>> alcost = std::make_shared<ALCost<n, m>>(prob, k);
    if (costs) {
      costs->emplace_back(alcost);
    }
    prob_al.SetCostFunction(std::move(alcost), k);
  }
  return prob_al;
}

}  // namespace augmented_lagrangian
}  // namespace altro