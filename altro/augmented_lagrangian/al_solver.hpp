// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief 使用增广拉格朗日处理任意约束的轨迹优化求解器，同时使用 DDP / iLQR 求解
 * 得到的无约束轨迹优化问题。
 *
 * @tparam n 编译时状态维度。
 * @tparam m 编译时控制维度。
 */
template <int n, int m>
class AugmentedLagrangianiLQR {
  template <class ConType>
  using ConstraintValueVec =
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>;

 public:
  explicit AugmentedLagrangianiLQR(int N) : ilqr_solver_(N), costs_(), max_violation_(N + 1) {}
  explicit AugmentedLagrangianiLQR(const problem::Problem& prob);

  void InitializeFromProblem(const problem::Problem& prob);

  /***************************** Getters **************************************/

  SolverStats& GetStats() { return ilqr_solver_.GetStats(); }
  const SolverStats& GetStats() const { return ilqr_solver_.GetStats(); }
  SolverOptions& GetOptions() { return ilqr_solver_.GetOptions(); }
  const SolverOptions& GetOptions() const { return ilqr_solver_.GetOptions(); }
  SolverStatus GetStatus() const { return status_; }
  std::shared_ptr<ALCost<n, m>> GetALCost(const int k) { return costs_.at(k); }
  ilqr::iLQR<n, m>& GetiLQRSolver() { return ilqr_solver_; }
  int NumSegments() const { return ilqr_solver_.NumSegments(); }

  int NumConstraints(const int& k) const;
  int NumConstraints() const;

  /**
   * @brief 打印问题中所有约束的摘要。
   * 
   * 打印每个约束的标签、其节点索引和违反向量。
   * 
   * 每个元素包含约束的基本描述、其所在的节点索引和其当前违反向量。
   * 
   * @param[in] should_sort 按约束的无穷范数对约束进行排序。默认为 false（不排序），
   *                        按节点索引排序。
   * @param[in] precision   控制数值输出的精度。
   */
  void PrintViolations(bool should_sort = false, int precision = 4) const {
    std::vector<constraints::ConstraintInfo> coninfo = GetConstraintInfo(should_sort);
    fmt::print("Got {} constraints\n", coninfo.size());
    for (const constraints::ConstraintInfo& info : coninfo) {
      fmt::print("{}\n", info.ToString(precision));
    }
  }

  /**
   * @brief 获取约束列表。
   * 
   * 每个元素包含约束的基本描述、其所在的节点索引和其当前违反向量。
   * 
   * @param should_sort 按约束违反的无穷范数对约束进行排序。
   * @return std::vector<constraints::ConstraintInfo>
   */
  std::vector<constraints::ConstraintInfo> GetConstraintInfo(bool should_sort = false) const {
    std::vector<constraints::ConstraintInfo> coninfo;
    int i_last = 0;
    for (int k = 0; k <= NumSegments(); ++k) {
      costs_[k]->GetConstraintInfo(&coninfo);
      for (int i = i_last; i < static_cast<int>(coninfo.size()); ++i) {
        coninfo[i].index = k;
      }
      i_last = coninfo.size();
    }
    if (should_sort) {
      auto comp = [](const constraints::ConstraintInfo& info1,
                     const constraints::ConstraintInfo& info2) {
        return info1.violation.lpNorm<Eigen::Infinity>() > info2.violation.lpNorm<Eigen::Infinity>();
      };
      std::sort(coninfo.begin(), coninfo.end(), comp);
    }
    return coninfo;
  }

  /***************************** Setters **************************************/

  /**
   * @brief 为所有约束和节点设置相同的惩罚参数。
   *
   * 要为不同约束和/或节点独立设置惩罚，请使用 GetALCost(k).SetPenalty<ConType>(rho, i)。
   *
   * @param rho
   */
  void SetPenalty(const double& rho);

  /**
   * @brief 为所有约束和节点设置相同的惩罚缩放参数。
   *
   * 要为不同约束和/或节点独立设置惩罚缩放，请使用 GetALCost(k).SetPenaltyScaling<ConType>(phi, i)。
   *
   * @param phi 惩罚参数 (phi > 1)。
   */
  void SetPenaltyScaling(const double& phi);

  /**
   * @brief 指定状态和控制轨迹的初始猜测。
   *
   * 此轨迹将被求解过程修改，并在求解完成后等于优化轨迹。
   *
   * @param traj 指向轨迹的指针。
   */
  void SetTrajectory(std::shared_ptr<Trajectory<n, m>> traj) {
    ilqr_solver_.SetTrajectory(std::move(traj));
  }

  /***************************** Methods **************************************/

  void Init();

  /**
   * @brief 使用 AL-iLQR 求解轨迹优化问题。
   *
   */
  void Solve();

  /**
   * @brief 更新所有约束的对偶变量
   *
   */
  void UpdateDuals();

  /**
   * @brief 更新所有约束的惩罚参数
   *
   */
  void UpdatePenalties();

  /**
   * @brief 计算增广拉格朗日的收敛准则
   *
   */
  void UpdateConvergenceStatistics();

  /**
   * @brief 检查求解是否可以终止。
   *
   * 将因为满足收敛准则或某种方式失败而终止。
   *
   * @return 如果求解器应该停止迭代则返回 true。
   */
  bool IsDone();

  /**
   * @brief 计算最大约束违反。
   * 通过评估增广拉格朗日代价来更新约束。
   *
   * @tparam p 用于计算违反的范数（默认为无穷）。
   * @param[in] Z 用于评估约束的轨迹。默认为内部 iLQR 求解器存储的轨迹。
   * @return 最大约束违反。如果求解成功应该接近零。
   */
  template <int p = Eigen::Infinity>
  double MaxViolation();

  template <int p = Eigen::Infinity>
  double MaxViolation(const Trajectory<n, m>& Z);

  /**
   * @brief 在不计算代价的情况下计算最大约束违反。
   * 将使用当前存储的约束值。
   *
   * @tparam p 用于计算违反的范数（默认为无穷）。
   * @return 最大约束违反。如果求解成功应该接近零。
   */
  template <int p = Eigen::Infinity>
  double GetMaxViolation();

  /**
   * @brief 获取所有约束和节点中使用的最大惩罚参数。
   *
   * @return 最大惩罚参数。
   */
  double GetMaxPenalty() const;

  void ResetDualVariables();

 private:
  Stopwatch CreateTimer(const std::string& name) { return GetStats().GetTimer()->Start(name); }

  ilqr::iLQR<n, m> ilqr_solver_;
  std::vector<std::shared_ptr<ALCost<n, m>>> costs_;
  SolverStatus status_ = SolverStatus::kUnsolved;
  VectorXd max_violation_;  // (N+1,) 每个节点的约束违反向量
};

////////////////////////////////////////////////////////////////////////////////
/**************************** Implementation **********************************/
////////////////////////////////////////////////////////////////////////////////

template <int n, int m>
AugmentedLagrangianiLQR<n, m>::AugmentedLagrangianiLQR(const problem::Problem& prob)
    : ilqr_solver_(prob.NumSegments()),
      costs_(),
      max_violation_(VectorXd::Zero(prob.NumSegments() + 1)) {
  InitializeFromProblem(prob);
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::InitializeFromProblem(const problem::Problem& prob) {
  max_violation_.setZero(prob.NumSegments() + 1);
  problem::Problem prob_al = BuildAugLagProblem<n, m>(prob, &costs_);
  ALTRO_ASSERT(static_cast<int>(costs_.size()) == prob.NumSegments() + 1,
               fmt::format("Got an incorrect number of cost functions. Expected {}, got {}",
                           prob.NumSegments(), costs_.size()));
  // ilqr_solver_.InitializeFromProblem(prob_al);
  ilqr_solver_.CopyFromProblem(prob_al, 0, prob.NumSegments() + 1);
  auto max_violation_callback = [this]() -> double { return this->GetMaxViolation(); };
  ilqr_solver_.SetConstraintCallback(max_violation_callback);
}

template <int n, int m>
int AugmentedLagrangianiLQR<n, m>::NumConstraints(const int& k) const {
  ALTRO_ASSERT(0 <= k && k <= NumSegments(),
               fmt::format("Invalid knot point index. Got {}, expected to be in range [{},{}]", k,
                           0, NumSegments()));
  ALTRO_ASSERT(
      static_cast<int>(costs_.size()) == NumSegments() + 1,
      "Cannot query the number of constraints before initializing the solver with a problem.");
  return costs_.at(k)->NumConstraints();
}

template <int n, int m>
int AugmentedLagrangianiLQR<n, m>::NumConstraints() const {
  int cnt = 0;
  for (int k = 0; k <= NumSegments(); ++k) {
    cnt += NumConstraints(k);
  }
  return cnt;
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::SetPenalty(const double& rho) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenalty<constraints::Equality>(rho);
    costs_[k]->template SetPenalty<constraints::Inequality>(rho);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::SetPenaltyScaling(const double& phi) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenaltyScaling<constraints::Equality>(phi);
    costs_[k]->template SetPenaltyScaling<constraints::Inequality>(phi);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::Init() {
  Stopwatch sw = CreateTimer("init");

  SolverStats& stats = GetStats();
  if (GetOptions().reset_duals) {
    ResetDualVariables();
  }
  if (GetOptions().initial_penalty > 0) {
    SetPenalty(GetOptions().initial_penalty);
  }
  stats.Reset();
  stats.Log("iter_al", 0);
  stats.Log("viol", MaxViolation());
  stats.Log("pen", GetMaxPenalty());
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::Solve() {
  // This check needs to happen before creating the first stopwatch
  GetOptions().profiler_enable ? GetStats().GetTimer()->Activate()
                               : GetStats().GetTimer()->Deactivate();
  Stopwatch sw = CreateTimer("al");

  Init();

  for (int iteration = 0; iteration < GetOptions().max_iterations_outer; ++iteration) {
    ilqr_solver_.Solve();
    UpdateDuals();
    UpdateConvergenceStatistics();

    // Print the log data here if iLQR isn't printing it
    bool is_ilqr_logging = GetStats().GetVerbosity() >= LogLevel::kInner;
    if (!is_ilqr_logging) {
      GetStats().PrintLast();
    }

    if (IsDone()) {
      break;
    }

    // If iLQR is printing the logs, print the header before every new AL iteration
    if (is_ilqr_logging) {
      GetStats().GetLogger().PrintHeader();
    }
    UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateDuals() {
  Stopwatch sw = CreateTimer("dual_update");

  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    // fmt::print("Updating Duals at index {}...\n", k);
    costs_[k]->UpdateDuals();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdatePenalties() {
  Stopwatch sw = CreateTimer("penalty_update");

  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    costs_[k]->UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateConvergenceStatistics() {
  Stopwatch sw = CreateTimer("stats");

  SolverStats& stats = GetStats();
  stats.iterations_outer++;
  stats.Log("viol", GetMaxViolation());
  stats.Log("pen", GetMaxPenalty());
  stats.Log("iter_al", stats.iterations_outer);
}

template <int n, int m>
bool AugmentedLagrangianiLQR<n, m>::IsDone() {
  Stopwatch sw = CreateTimer("convergence_check");

  SolverStats& stats = GetStats();
  SolverOptions& opts = GetOptions();
  const bool are_constraints_satisfied = stats.violations.back() < opts.constraint_tolerance;
  const bool is_max_penalty_exceeded = stats.max_penalty.back() > opts.maximum_penalty;
  const bool is_max_outer_iterations_exceeded = stats.iterations_outer >= opts.max_iterations_outer;
  const bool is_max_total_iterations_exeeded = stats.iterations_total >= opts.max_iterations_total;
  if (ilqr_solver_.GetStatus() != SolverStatus::kSolved) {
    status_ = ilqr_solver_.GetStatus();
    return true;
  }
  if (are_constraints_satisfied) {
    if (ilqr_solver_.GetStatus() == SolverStatus::kSolved) {
      status_ = SolverStatus::kSolved;
      return true;
    }
  }
  if (is_max_penalty_exceeded) {
    status_ = SolverStatus::kMaxPenalty;
    return true;
  }
  if (is_max_outer_iterations_exceeded) {
    status_ = SolverStatus::kMaxOuterIterations;
    return true;
  }
  if (is_max_total_iterations_exeeded) {
    status_ = SolverStatus::kMaxIterations;
    return true;
  }
  return false;
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::MaxViolation() {
  ilqr_solver_.Cost();  // Calculate cost to update constraints
  return GetMaxViolation<p>();
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::MaxViolation(const Trajectory<n, m>& Z) {
  ilqr_solver_.Cost(Z);  // Calculate cost to update constraints
  return GetMaxViolation<p>();
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::GetMaxViolation() {
  for (int k = 0; k <= NumSegments(); ++k) {
    max_violation_(k) = costs_[k]->template MaxViolation<p>();
  }
  return max_violation_.template lpNorm<p>();
}

template <int n, int m>
double AugmentedLagrangianiLQR<n, m>::GetMaxPenalty() const {
  double max_penalty = 0.0;

  for (int k = 0; k <= NumSegments(); ++k) {
    max_penalty = std::max(max_penalty, costs_[k]->MaxPenalty());
  }
  return max_penalty;
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::ResetDualVariables() {
  for (auto& alcost : costs_) {
    alcost->ResetDualVariables();
  }
}

}  // namespace augmented_lagrangian
}  // namespace altro