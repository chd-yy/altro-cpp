// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <array>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <thread>

#include "altro/common/solver_stats.hpp"
#include "altro/common/state_control_sized.hpp"
#include "altro/common/threadpool.hpp"
#include "altro/common/timer.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace ilqr {

/**
 * @brief 使用迭代 LQR 求解无约束轨迹优化问题。
 *
 * 该类可以默认构造或为给定数量的节点初始化。目前，一旦设置，
 * 节点数量就无法更改。如果默认初始化，则在第一次调用
 * `CopyFromProblem` 时从问题中提取节点数量。
 *
 * iLQR 算法通过对代价函数进行二阶近似和对动力学进行一阶展开来工作。
 * 然后围绕当前最优轨迹的估计构造局部最优的反馈控制策略，
 * 该策略在"后向传递"期间使用时变 LQR 的推广来计算。
 * 然后在"前向传递"期间使用该策略模拟系统前进，并重复该过程直到收敛。
 * 由于系统在每次迭代中都向前模拟，iLQR 实际上只直接优化控制变量。
 *
 * @tparam n 编译时状态维度。
 * @tparam m 编译时控制维度。
 */
template <int n = Eigen::Dynamic, int m = Eigen::Dynamic>
class iLQR {
 public:
  explicit iLQR(int N) : N_(N), knotpoints_() { ResetInternalVariables(); }
  explicit iLQR(const problem::Problem& prob)
      : N_(prob.NumSegments()), initial_state_(std::move(prob.GetInitialStatePointer())) {
    InitializeFromProblem(prob);
  }

  iLQR(const iLQR& other) = delete;
  iLQR& operator=(const iLQR& other) = delete;
  iLQR(iLQR&& other) noexcept : N_(other.N_),
                                initial_state_(std::move(other.initial_state_)),
                                stats_(std::move(other.stats_)),
                                knotpoints_(std::move(other.knotpoints_)),
                                Z_(std::move(other.Z_)),
                                Zbar_(std::move(other.Zbar_)),
                                status_(other.status_),
                                costs_(std::move(other.costs_)),
                                grad_(std::move(other.grad_)),
                                rho_(other.rho_),
                                drho_(other.drho_),
                                deltaV_(std::move(other.deltaV_)),
                                is_initial_state_set(other.is_initial_state_set),
                                max_violation_callback_(std::move(other.max_violation_callback_)) {}

  /**
   * @brief 将 Problem 类的数据复制到 iLQR 求解器中
   *
   * 捕获每个节点的代价和动力学对象的共享指针，
   * 将它们存储在相应的 KnotPointFunctions 对象中。
   *
   * 假设问题和求解器都具有相同数量的节点。
   *
   * 允许复制节点的子集，因为将来此方法可能用于为混合/切换动力学
   * 指定编译时大小。
   *
   * 将节点追加到求解器中当前的节点。
   *
   * 从问题中捕获初始状态作为共享指针，因此通过修改原始问题的
   * 初始状态来更改求解器的初始状态。
   *
   * @tparam n2 编译时状态维度。可以是 Eigen::Dynamic (-1)
   * @tparam m2 编译时控制维度。可以是 Eigen::Dynamic (-1)
   * @param prob 轨迹优化问题
   * @param k_start 要复制数据的起始索引（包含）。0 <= k_start < N+1
   * @param k_stop 要复制数据的终止索引（不包含）。0 < k_stop <= N+1
   */
  template <int n2 = n, int m2 = m>
  void CopyFromProblem(const problem::Problem& prob, int k_start, int k_stop) {
    ALTRO_ASSERT(0 <= k_start && k_start <= N_,
                 fmt::format("Start index must be in the interval [0,{}]", N_));
    ALTRO_ASSERT(0 <= k_stop && k_stop <= N_ + 1,
                 fmt::format("Start index must be in the interval [0,{}]", N_ + 1));
    ALTRO_ASSERT(prob.IsFullyDefined(), "Expected problem to be fully defined.");
    for (int k = k_start; k < k_stop; ++k) {
      if (n != Eigen::Dynamic) {
        ALTRO_ASSERT(
            prob.GetDynamics(k)->StateDimension() == n,
            fmt::format("Inconsistent state dimension at knot point {}. Expected {}, got {}", k, n,
                        prob.GetDynamics(k)->StateDimension()));
      }
      if (m != Eigen::Dynamic) {
        ALTRO_ASSERT(
            prob.GetDynamics(k)->ControlDimension() == m,
            fmt::format("Inconsistent control dimension at knot point {}. Expected {}, got {}", k,
                        m, prob.GetDynamics(k)->ControlDimension()));
      }
      std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(k);
      std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(k);
      knotpoints_.emplace_back(std::make_unique<ilqr::KnotPointFunctions<n2, m2>>(model, costfun));
    }
    initial_state_ = prob.GetInitialStatePointer();
    is_initial_state_set = true;
  }

  template <int n2 = n, int m2 = m>
  void InitializeFromProblem(const problem::Problem& prob) {
    ALTRO_ASSERT(prob.NumSegments() == N_,
                 fmt::format("Number of segments in problem {}, should be equal to the number of "
                             "segments in the solver, {}",
                             prob.NumSegments(), N_));
    CopyFromProblem<n2, m2>(prob, 0, N_ + 1);
    ResetInternalVariables();
  }

  /***************************** 获取器 **************************************/
  /**
   * @brief 获取轨迹的指针
   *
   */
  std::shared_ptr<Trajectory<n, m>> GetTrajectory() { return Z_; }

  /**
   * @brief 返回轨迹中的段数
   */
  int NumSegments() const { return N_; }
  /**
   * @brief 获取节点函数对象，该对象包含每个节点的所有数据，
   * 包括代价和动力学展开、反馈和前馈增益、代价到达展开等。
   *
   * @param k 节点索引，0 <= k <= N_
   * @return KnotPointFunctions 类的引用
   */
  KnotPointFunctions<n, m>& GetKnotPointFunction(int k) {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    return *(knotpoints_[k]);
  }

  SolverStats& GetStats() { return stats_; }
  const SolverStats& GetStats() const { return stats_; }
  SolverOptions& GetOptions() { return stats_.GetOptions(); }
  const SolverOptions& GetOptions() const { return stats_.GetOptions(); }
  VectorXd& GetCosts() { return costs_; }
  SolverStatus GetStatus() const { return status_; }
  std::shared_ptr<VectorXd> GetInitialState() { return initial_state_; }
  double GetRegularization() { return rho_; }

  /**
   * @brief 获取轨迹分配到任务的分配方案。
   *
   * 任务由一组连续的节点索引定义，其展开将被串行处理。
   * 尽管所有节点都可以并行处理，但通常最好将轨迹"分块"为
   * 可用并行处理器的数量。
   *
   * 大多数用户不需要使用此信息。
   *
   * @return std::vector<int>& 严格递增的节点索引向量。
   * 每个任务处理区间 [`inds[k]`, `inds[k+1]`) 中的节点，
   * 其中 `inds[0] = 0` 且 inds.back() = N+1`。
   *
   */
  std::vector<int>& GetTaskAssignment() {
    if (ShouldRedoTaskAssignment()) {
      DefaultTaskAssignment();
    }
    return work_inds_;
  }

  /**
   * @brief 获取 iLQR 求解器中使用的线程数
   *
   * @return 线程数
   */
  size_t NumThreads() const { return pool_.NumThreads(); }

  /**
   * @brief 获取可以并行执行的任务数。
   *
   * 通过 `AssignWork` 控制。
   *
   */
  int NumTasks() const { return work_inds_.size() - 1; }

  /**
   * @brief 创建一个新的零初始化轨迹。
   *
   * 假设使用均匀的时间步长。
   * 轨迹自动链接到求解器，并在求解期间和求解后用作初始猜测
   * 和优化解的存储位置。
   *
   * @param dt 轨迹中使用的时间步长。
   * @return std::shared_ptr<Trajectory<n, m>> 一个新的零初始化轨迹。
   */
  std::shared_ptr<Trajectory<n, m>> MakeTrajectory(float dt) {
    Z_ = std::make_shared<Trajectory<n, m>>(NumSegments());
    Z_->SetUniformStep(dt);
    return Z_;
  }

  /***************************** 设置器 **************************************/
  /**
   * @brief 存储轨迹的指针
   *
   * 该轨迹将用作初始猜测，也将是优化轨迹的存储位置。
   *
   * @param traj 轨迹的指针
   */
  void SetTrajectory(std::shared_ptr<Trajectory<n, m>> traj) {
    Z_ = std::move(traj);
    Zbar_ = std::make_unique<Trajectory<n, m>>(*Z_);
    Zbar_->SetZero();
  }

  void SetConstraintCallback(const std::function<double()>& max_violation) {
    max_violation_callback_ = max_violation;
  }

  /**
   * @brief 设置节点索引到可并行化任务的划分。
   *
   * 定义应作为单个任务串行处理的连续节点组。
   * 然后每个组可以独立并行运行。
   * 为获得最佳性能，任务数应等于可用核心数。
   *
   * 一旦设置，如果请求的线程数（通过 `GetOptions().nthreads`）
   * 或每个线程的任务数（通过 `GetOptions().tasks_per_thread`）发生变化，
   * 求解器将不再自动调整任务数。一旦设置，用户有责任根据需要修改此值。
   *
   * @param inds 严格递增的节点索引向量。对于长度为 N 的向量，
   * 它定义了 N-1 个任务，其中每个任务处理区间 [`inds[i]`, `inds[i+1]`) 中的索引。
   */
  void SetTaskAssignment(std::vector<int> inds) {
    ALTRO_ASSERT(work_inds_.back() == NumSegments() + 1,
                 "Work inds should include the terminal index.");
    ALTRO_ASSERT(work_inds_[0] == 0, "Work inds should start with a 0.");
    ALTRO_ASSERT(work_inds_.size() >= 2, "Work inds must have at least 2 elements.");
    bool is_sorted = true;
    for (int i = 1; i < inds.size(); ++i) {
      if (inds[i] <= inds[i - 1]) {
        is_sorted = false;
      }
    }
    ALTRO_ASSERT(is_sorted, "Work inds must be a set of strictly increasing integers.");
    work_inds_ = std::move(inds);
    custom_work_assignment_ = true;
  }

  /***************************** 算法 **************************************/
  /**
   * @brief 使用 iLQR 求解轨迹优化问题
   *
   * @post 提供的轨迹将被局部最优的动态可行轨迹覆盖。
   * 通过 GetStatus() 和 GetStats() 获得的求解器状态和统计信息会被更新。
   * 如果 `GetStatus == SolverStatus::kSuccess`，则求解成功。
   *
   */
  void Solve() {
    ALTRO_ASSERT(is_initial_state_set, "Initial state must be set before solving.");
    ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");

    // TODO(bjackson): 允许求解器优化更长轨迹的一部分？
    ALTRO_ASSERT(Z_->NumSegments() == N_,
                 fmt::format("Initial trajectory must have length {}", N_));

    // 启动分析器
    GetOptions().profiler_enable ? stats_.GetTimer()->Activate() : stats_.GetTimer()->Deactivate();
    Stopwatch sw = stats_.GetTimer()->Start("ilqr");

    SolveSetup();  // 重置所有内部变量
    Rollout();     // 使用初始控制模拟系统前进
    stats_.initial_cost = Cost();

    for (int iter = 0; iter < GetOptions().max_iterations_inner; ++iter) {
      UpdateExpansions();
      BackwardPass();
      ForwardPass();
      UpdateConvergenceStatistics();

      if (stats_.GetVerbosity() >= LogLevel::kInner) {
        stats_.PrintLast();
      }

      if (IsDone()) {
        break;
      }
    }

    WrapUp();
  }

  /**
   * @brief 计算当前轨迹的代价
   *
   * 默认情况下，它将使用求解器中存储的当前猜测，
   * 但也可以传递任何兼容的轨迹。
   *
   * @return double 当前代价
   */
  double Cost() {
    ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");
    return Cost(*Z_);
  }
  double Cost(const Trajectory<n, m>& Z) {
    Stopwatch sw = stats_.GetTimer()->Start("cost");
    CalcIndividualCosts(Z);
    return costs_.sum();
  }

  /**
   * @brief 更新代价和动力学展开
   *
   * 注意：还会计算每个节点的代价。
   *
   * 计算代价和动力学的一阶和二阶展开，
   * 将结果存储在每个节点的 KnotPointFunctions 类中。
   *
   * @pre 轨迹必须设置为最优轨迹的下一个猜测。
   * 轨迹不能为 nullptr，必须通过 SetTrajectory 设置。
   *
   * @post knotpoints_[k] 的展开已更新，0 <= k < N_
   *
   */
  void UpdateExpansions() {
    Stopwatch sw = stats_.GetTimer()->Start("expansions");
    ALTRO_ASSERT(Z_ != nullptr, "Trajectory pointer must be set before updating the expansions.");

    int nthreads = NumThreads();
    if (nthreads <= 1) {
      UpdateExpansionsBlock(0, NumSegments() + 1);
    } else {
      {
        Stopwatch sw2 = stats_.GetTimer()->Start("add_tasks");
        for (const std::function<void()>& task : tasks_) {
          pool_.AddTask(task);
        }
      }
      pool_.Wait();
    }
  }

  /**
   * @brief 计算局部最优的线性反馈策略
   *
   * 后向传递使用时变 LQR 来计算最优的线性反馈控制策略。
   * 随着求解收敛，常数前馈项应该趋于零。
   * 求解还计算代价到达的局部二次近似。
   *
   * @pre 已使用 UpdateExpansions 计算代价和动力学展开。
   *
   * @post 每个节点的 KnotPointFunctions 类中的前馈和反馈增益、
   * 动作值展开和代价到达展开项都已更新。
   * 总体预期代价减少存储在 deltaV_ 中。
   *
   */
  void BackwardPass() {
    Stopwatch sw = stats_.GetTimer()->Start("backward_pass");

    // Regularization
    Eigen::ComputationInfo info;

    // Terminal Cost-to-go
    knotpoints_[N_]->CalcTerminalCostToGo();
    Eigen::Matrix<double, n, n>* Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
    Eigen::Matrix<double, n, 1>* Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

    int max_reg_count = 0;
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;

    bool repeat_backwardpass = true;
    while (repeat_backwardpass) {
      for (int k = N_ - 1; k >= 0; --k) {
        // TODO(bjackson)[SW-16103] Create a test that checks this
        knotpoints_[k]->CalcActionValueExpansion(*Sxx_prev, *Sx_prev);
        knotpoints_[k]->RegularizeActionValue(rho_);
        info = knotpoints_[k]->CalcGains();

        // Handle solve failure
        if (info != Eigen::Success) {

          IncreaseRegularization();

          // Reset the cost-to-go pointers to the terminal expansion
          Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
          Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

          // Check if we're at max regularization
          if (rho_ >= GetOptions().bp_reg_max) {
            max_reg_count++;
          }

          if (max_reg_count >= GetOptions().bp_reg_fail_threshold) {
            status_ = SolverStatus::kBackwardPassRegularizationFailed;
            repeat_backwardpass = false;
          }
          break;
        }

        // Update Cost-To-Go
        knotpoints_[k]->CalcCostToGo();
        knotpoints_[k]->AddCostToGo(&deltaV_);

        Sxx_prev = &(knotpoints_[k]->GetCostToGoHessian());
        Sx_prev = &(knotpoints_[k]->GetCostToGoGradient());

        // Backward pass successful if it calculates the cost to go at
        // the first knot point.
        if (k == 0) {
          repeat_backwardpass = false;
        }
      } // end for
    } // end while
    stats_.Log("reg", rho_);
    DecreaseRegularization();
  }

  /**
   * @brief Simulate the dynamics forward from the initial state
   *
   * By default it will simulate the system forward open-loop.
   *
   */
  void Rollout() {
    Z_->State(0) = *initial_state_;
    for (int k = 0; k < N_; ++k) {
      knotpoints_[k]->Dynamics(Z_->State(k), Z_->Control(k), Z_->GetTime(k), Z_->GetStep(k),
                               Z_->State(k + 1));
    }
  }

  /**
   * @brief Simulate the system forward using the feedback and feedforward
   * gains calculated during the backward pass.
   *
   * @param alpha Line search parameter, 0 < alpha <= 1.
   * @return true If the the state and control bounds are not violated.
   */
  bool RolloutClosedLoop(const double alpha) {
    Stopwatch sw = stats_.GetTimer()->Start("rollout");

    Zbar_->State(0) = *initial_state_;
    for (int k = 0; k < N_; ++k) {
      MatrixNxMd<m, n>& K = GetKnotPointFunction(k).GetFeedbackGain();
      VectorNd<m>& d = GetKnotPointFunction(k).GetFeedforwardGain();

      // TODO(bjackson): Make this a function of the dynamics
      VectorNd<n> dx = Zbar_->State(k) - Z_->State(k);
      Zbar_->Control(k) = Z_->Control(k) + K * dx + d * alpha;

      // Simulate forward with feedback
      GetKnotPointFunction(k).Dynamics(Zbar_->State(k), Zbar_->Control(k), Zbar_->GetTime(k),
                                       Zbar_->GetStep(k), Zbar_->State(k + 1));

      if (GetOptions().check_forwardpass_bounds) {
        if (Zbar_->State(k + 1).norm() > GetOptions().state_max) {
          // TODO(bjackson): Emit warning (need logging mechanism)
          status_ = SolverStatus::kStateLimit;
          return false;
        }
        if (Zbar_->Control(k).norm() > GetOptions().control_max) {
          // TODO(bjackson): Emit warning (need logging mechanism)
          status_ = SolverStatus::kControlLimit;
          return false;
        }
      }
    }
    status_ = SolverStatus::kUnsolved;
    return true;
  }

  /**
   * @brief Attempt to find a better state-control trajectory
   *
   * Using the feedback policy computed during the backward pass,
   * simulate the system forward and make sure the resulting trajectory
   * decreases the overall cost and make sufficient progress towards a
   * local minimum (via pseudo Wolfe conditions).
   *
   * @post The current trajectory candidate Z_ is updated with the new guess.
   *
   */
  void ForwardPass() {
    Stopwatch sw = stats_.GetTimer()->Start("forward_pass");
    SolverOptions& opts = GetOptions();

    double J0 = costs_.sum();  // Calculated during UpdateExpansions

    double alpha = 1.0;
    double z = -1.0;
    int iter_fp = 0;
    bool success = false;

    double J = J0;

    for (; iter_fp < opts.line_search_max_iterations; ++iter_fp) {
      if (RolloutClosedLoop(alpha)) {
        J = Cost(*Zbar_);
        double expected = -alpha * (deltaV_[0] + alpha * deltaV_[1]);
        if (expected > 0.0) {
          z = (J0 - J) / expected;
        } else {
          z = -1.0;
        }

        if (opts.line_search_lower_bound <= z && z <= opts.line_search_upper_bound && J < J0) {
          success = true;
          // stats_.improvement_ratio.emplace_back(z);
          stats_.Log("cost", J);
          stats_.Log("alpha", alpha);
          stats_.Log("z", z);
          break;
        }
      }
      alpha /= opts.line_search_decrease_factor;
    }

    if (success) {
      (*Z_) = (*Zbar_);
    } else {
      IncreaseRegularization();
      J = J0;
    }

    if (J > J0) {
      // TODO(bjackson): Emit warning (needs logging)
      status_ = SolverStatus::kCostIncrease;
    }
  }

  /**
   * @brief Evaluate all the information necessary to check convergence
   *
   * Calculates the gradient, change in cost, etc. Updates the solver statistics
   * accordingly.
   *
   * @post Increments the number of solver iterations
   */
  void UpdateConvergenceStatistics() {
    Stopwatch sw = stats_.GetTimer()->Start("stats");

    double dgrad = NormalizedFeedforwardGain();
    double dJ = 0.0;
    if (stats_.iterations_inner == 0) {
      dJ = stats_.initial_cost - stats_.cost.back();
    } else {
      dJ = stats_.cost.rbegin()[1] - stats_.cost.rbegin()[0];
    }

    // stats_.gradient.emplace_back(dgrad);
    stats_.iterations_inner++;
    stats_.iterations_total++;
    stats_.Log("dJ", dJ);
    stats_.Log("viol", max_violation_callback_());
    stats_.Log("iters", stats_.iterations_total);
    stats_.Log("grad", dgrad);
    stats_.NewIteration();
  }

  /**
   * @brief Checks if the solver is done solving and can stop iterating
   *
   * The solver can exit because it has successfully converged or because it
   * has entered a bad state and needs to exit.
   *
   * @return true If the solver should stop iterating
   */
  bool IsDone() {
    Stopwatch sw = stats_.GetTimer()->Start("convergence_check");
    SolverOptions& opts = GetOptions();

    bool cost_decrease = stats_.cost_decrease.back() < opts.cost_tolerance;
    bool gradient = stats_.gradient.back() < opts.gradient_tolerance;
    bool is_done = false;

    if (cost_decrease && gradient) {
      status_ = SolverStatus::kSolved;
      is_done = true;
    } else if (stats_.iterations_inner >= opts.max_iterations_inner) {
      status_ = SolverStatus::kMaxInnerIterations;
      is_done = true;
    } else if (stats_.iterations_total >= opts.max_iterations_total) {
      status_ = SolverStatus::kMaxIterations;
      is_done = true;
    } else if (status_ != SolverStatus::kUnsolved) {
      is_done = true;
    }

    return is_done;
  }

  /**
   * @brief Initialize the solver to pre-compute any needed information and
   * be ready for a solve.
   *
   * This method should ensure the solver enters a reproducible state prior
   * to each solve, so that the `Solve()` method can be called multiple times.
   *
   */
  void SolveSetup() {
    Stopwatch sw = stats_.GetTimer()->Start("init");
    stats_.iterations_inner = 0;
    stats_.SetVerbosity(GetOptions().verbose);

    // Make sure Zbar has the same times as the initial trajectory
    if (Z_ != nullptr) {
      int k;
      for (k = 0; k < N_; ++k) {
        Zbar_->SetStep(k, Z_->GetStep(k));
        Zbar_->SetTime(k, Z_->GetTime(k));
      }
      Zbar_->SetTime(N_, Z_->GetTime(N_));
    }

    ResetInternalVariables();
  }

  /**
   * @brief Perform any operations needed to return the solver to a desireable
   * state after the iterations have stopped.
   *
   */
  void WrapUp() {}

  /**
   * @brief Calculate the infinity-norm of the feedforward gains, normalized
   * by the current control values.
   *
   * Provides an approximation to the gradient of the Lagrangian.
   *
   * @return double
   */
  double NormalizedFeedforwardGain() {
    for (int k = 0; k < N_; ++k) {
      VectorNd<m>& d = GetKnotPointFunction(k).GetFeedforwardGain();
      grad_(k) = (d.array().abs() / (Z_->Control(k).array().abs() + 1)).maxCoeff();
    }
    return grad_.sum() / grad_.size();
  }

  void UpdateExpansionsBlock(int start, int stop) {
    for (int k = start; k < stop; ++k) {
      KnotPoint<n, m>& z = Z_->GetKnotPoint(k);
      knotpoints_[k]->CalcCostExpansion(z.State(), z.Control());
      knotpoints_[k]->CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(), z.GetStep());
      costs_(k) = GetKnotPointFunction(k).Cost(z.State(), z.Control());
    }
  }

 private:
  void ResetInternalVariables() {
    status_ = SolverStatus::kUnsolved;
    costs_ = VectorXd::Zero(N_ + 1);
    grad_ = VectorXd::Zero(N_);
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;
    rho_ = GetOptions().bp_reg_initial;
    drho_ = 0.0;

    LaunchThreads();
  }

  /**
   * @brief Check if the tasks need to re-assigned.
   *
   * Will not overrite tasks once they have been assigned manually via
   * `SetTaskAssignment()`.
   *
   */
  bool ShouldRedoTaskAssignment() const {
    bool is_custom = custom_work_assignment_;
    int tasks_per_thread = GetOptions().tasks_per_thread;
    bool has_expected_number_of_tasks = NumTasks() == GetOptions().NumThreads() * tasks_per_thread;
    bool keep_assignment = is_custom || has_expected_number_of_tasks;
    return !keep_assignment;
  }

  void LaunchThreads() {
    size_t nthreads = GetOptions().NumThreads();

    // Reset the thread pool if the requested number of threads changes
    int threadpool_size = NumThreads();
    bool single_threaded = threadpool_size == 0 && nthreads == 1;
    if (single_threaded) {
      return;
    }
    bool num_threads_changed = nthreads != NumThreads();
    if (N_ > 0 && (num_threads_changed || ShouldRedoTaskAssignment())) {
      if (pool_.IsRunning()) {
        pool_.StopThreads();
      }
      tasks_.clear();

      // Create tasks
      std::vector<int>& work_inds = GetTaskAssignment();
      int ntasks = NumTasks();
      for (int i = 0; i < ntasks; ++i) {
        int start = work_inds[i];
        int stop = work_inds[i + 1];
        auto expansion_block = [this, start, stop]() { UpdateExpansionsBlock(start, stop); };
        tasks_.emplace_back(std::move(expansion_block));
      }

      // Start the pool
      if (nthreads > 1) {
        pool_.LaunchThreads(nthreads);
      }
    }
  }

  void DefaultTaskAssignment() {
    int nthreads = GetOptions().NumThreads();
    int ntasks = nthreads * GetOptions().tasks_per_thread;
    double step = NumSegments() / static_cast<double>(ntasks);
    work_inds_.clear();
    for (double val = 0.0; val <= NumSegments(); val += step) {
      work_inds_.emplace_back(static_cast<int>(round(val)));
    }
    ALTRO_ASSERT(work_inds_.back() == NumSegments(),
                 "Work inds should include the terminal index.");
    work_inds_.back() += 1;  // Increment the last index to include the terminal index.
  }

  /**
   * @brief Calculate the cost of each individual knot point
   *
   * @param Z
   */
  void CalcIndividualCosts(const Trajectory<n, m>& Z) {
    // TODO(bjackson): do this in parallel
    for (int k = 0; k <= N_; ++k) {
      costs_(k) = GetKnotPointFunction(k).Cost(Z.State(k), Z.Control(k));
    }
  }

  /**
   * @brief Increase the regularization, steering the steps closer towards
   * gradient descent (more robust, less efficient).
   *
   */
  void IncreaseRegularization() {
    const SolverOptions& opts = GetOptions();
    drho_ = std::max(drho_ * opts.bp_reg_increase_factor, opts.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
    rho_ = std::min(rho_, opts.bp_reg_max);
  }

  /**
   * @brief Decrease the regularization term.
   *
   */
  void DecreaseRegularization() {
    const SolverOptions& opts = GetOptions();
    drho_ = std::min(drho_ / opts.bp_reg_increase_factor, 1 / opts.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
    rho_ = std::min(rho_, opts.bp_reg_max);
  }

  int N_;  // number of segments
  std::shared_ptr<VectorXd> initial_state_;
  SolverStats stats_;  // solver statistics (iterations, cost at each iteration, etc.)

  // TODO(bjackson): Create a non-templated base class to allow different dimensions.
  std::vector<std::unique_ptr<KnotPointFunctions<n, m>>>
      knotpoints_;                          // problem description and data
  std::shared_ptr<Trajectory<n, m>> Z_;     // current guess for the trajectory
  std::unique_ptr<Trajectory<n, m>> Zbar_;  // temporary trajectory for forward pass

  SolverStatus status_ = SolverStatus::kUnsolved;

  VectorXd costs_;     // costs at each knot point
  VectorXd grad_;      // gradient at each knot point
  double rho_ = 0.0;   // regularization
  double drho_ = 0.0;  // regularization derivative (damping)
  std::array<double, 2> deltaV_;

  bool is_initial_state_set = false;
  bool custom_work_assignment_ = false;  // Has user assigned a custom task assignment

  std::function<double()> max_violation_callback_ = []() { return 0.0; };
  std::vector<std::function<void()>> tasks_;
  std::vector<int> work_inds_;
  ThreadPool pool_;
};

}  // namespace ilqr
}  // namespace altro