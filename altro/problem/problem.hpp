// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/format.h>
#include <memory>
#include <vector>

#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace problem {

/**
 * @brief 简单的离散动力学，占位用途，不进行任何计算。
 *
 * 用于最后一个时间步，向后续流程提供状态维度信息。
 *
 */
class IdentityDynamics : public DiscreteDynamics {
 public:
  using DiscreteDynamics::Evaluate;
  explicit IdentityDynamics(int n, int m) : n_(n), m_(m) {
    ALTRO_ASSERT(n > 0, "State dimension must be greater than zero.");
    ALTRO_ASSERT(m > 0, "Control dimension must be greater than zero.");
  }

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& /*u*/, const float /*t*/,
                const float /*h*/, Eigen::Ref<VectorXd> xnext) override {
    xnext = x;
  }
  void Jacobian(const VectorXdRef& /*x*/, const VectorXdRef& /*u*/, const float /*t*/,
                const float /*h*/, Eigen::Ref<MatrixXd> jac) override {
    jac.setIdentity();
  }
  void Hessian(const VectorXdRef& /*x*/, const VectorXdRef& /*u*/, const float /*t*/,
               const float /*h*/, const VectorXdRef& /*b*/, Eigen::Ref<MatrixXd> hess) override {
    hess.setZero();
  }
  bool HasHessian() const override { return true; }

 private:
  int n_;
  int m_;
};

/**
 * @brief Describes and evaluates the trajectory optimization problem
 *
 * Describes generic trajectory optimization problems of the following form:
 * minimize    sum( J(X[k], U[k]), k = 0:N )
 *   X,U
 * subject to f_k(X[k], U[k], X[k+1], U[k+1]) = 0, k = 0:N-1
 *            g_k(X[k], U[k]) = 0,                 k = 0:N
 *            h_ki(X[k], U[k]) in cone K_i,        k = 0:N, i = 0...
 */
class Problem {
  template <class ConType>
  using ConstraintSet = std::vector<constraints::ConstraintPtr<ConType>>;

 public:
  /**
   * @brief 使用 N 个区段初始化一个新的问题
   *
   * @param N 轨迹区段数量（等于结点数减 1）
   */
  explicit Problem(const int N, std::shared_ptr<VectorXd> initial_state = std::make_shared<VectorXd>(0))
      : N_(N), initial_state_(initial_state), costfuns_(N + 1, nullptr), models_(N + 1, nullptr), eq_(N + 1), ineq_(N + 1) {}

  Problem& operator=(const Problem& other) {
    *initial_state_ = *(other.initial_state_);
    costfuns_ = other.costfuns_;
    models_ = other.models_;
    eq_ = other.eq_;
    ineq_ = other.ineq_;
    return *this;
  }

  Problem(const Problem& other)
      : costfuns_(other.costfuns_), models_(other.models_), eq_(other.eq_), ineq_(other.ineq_) {
    *initial_state_ = *(other.initial_state_);
  }

  Problem& operator=(Problem&& other) {
    *initial_state_ = *(other.initial_state_);
    costfuns_ = std::move(other.costfuns_);
    models_ = std::move(other.models_);
    eq_ = std::move(other.eq_);
    ineq_ = std::move(other.ineq_);
    return *this;
  }

  Problem(Problem&& other)
      : initial_state_(other.initial_state_),
        costfuns_(std::move(other.costfuns_)),
        models_(std::move(other.models_)),
        eq_(std::move(other.eq_)),
        ineq_(std::move(other.ineq_)) {}
  /**
   * @brief 设置问题的初始状态
   *
   * @param x0 初始状态
   */
  void SetInitialState(const VectorXdRef& x0) { *initial_state_ = x0; }

  /**
   * @brief 设置第 k 个结点的代价函数
   *
   * @param costfun 代价函数对象指针
   * @param k 结点索引（0 <= k <= N）
   */
  void SetCostFunction(std::shared_ptr<CostFunction> costfun, int k) {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    costfuns_[k] = std::move(costfun);
  }

  /**
   * @brief 为一段连续的结点区间设置代价函数
   *
   * 一般地，为避免在按结点并行时产生竞争，输入向量的每个元素应指向独立对象。
   *
   * @tparam CostFun 继承自 `CostFunction` 的类型。
   * @param costfuns 代价函数指针的向量。这些指针会被直接复制到问题与求解器中。
   * 并行时需要用户自行确保不会产生数据竞争；通常应为每个结点创建独立的代价函数实例。
   * @param k_start 起始索引（含）。从该结点开始依次复制；默认从轨迹起点开始。
   */
  template <class CostFun>
  void SetCostFunction(const std::vector<std::shared_ptr<CostFun>>& costfuns, int k_start = 0) {
    for (size_t i = 0; i < costfuns.size(); ++i) {
      int k = i + k_start;
      SetCostFunction(costfuns[i], k);
    }
  }

  /**
   * @brief 设置第 k 个时间步的动力学模型
   *
   * @param model 动力学函数对象指针
   * @param k 时间步（0 <= k < N）
   */
  void SetDynamics(std::shared_ptr<DiscreteDynamics> model, int k) {
    ALTRO_ASSERT(model != nullptr, "Cannot pass a nullptr for the dynamics.");
    ALTRO_ASSERT((k >= 0) && (k < N_), "Invalid knot point index.");

    // 在最后一个时间步创建一个占位动力学模型，用于提供状态与控制维度
    if (k == N_ - 1) {
      models_.at(N_) =
          std::make_shared<IdentityDynamics>(model->StateDimension(), model->ControlDimension());
    }
    models_[k] = std::move(model);
  }

  /**
   * @brief 为一段连续的结点区间设置动力学函数
   *
   * 一般地，为避免在按结点并行时产生竞争，输入向量的每个元素应指向独立对象。
   *
   * @tparam Dynamics 继承自 `problem::DiscreteDynamics` 的类型。
   * @param models 动力学函数指针的向量。这些指针会被直接复制到问题与求解器中。
   * 并行时需要用户自行确保不会产生数据竞争；通常应为每个结点创建独立的动力学实例。
   * 对于 `DiscretizedModel` 尤其重要，因为其会为数值积分分配临时存储，必须为每个结点
   * 创建独立模型才能保证线程安全。
   * @param k_start 起始索引（含）。从该结点开始依次复制；默认从轨迹起点开始。
   */
  template <class Dynamics>
  void SetDynamics(const std::vector<std::shared_ptr<Dynamics>>& models, int k_start = 0) {
    for (size_t i = 0; i < models.size(); ++i) {
      int k = i + k_start;
      SetDynamics(models[i], k);
    }
  }

  template <class ConstraintObject>
  void SetConstraint(std::shared_ptr<ConstraintObject> con, int k) {
    using ConType = typename ConstraintObject::ConstraintType;
    constraints::ConstraintPtr<ConType> ptr = con;
    SetConstraint<ConType>(ptr, k);
  }

  template <class ConType>
  void SetConstraint(std::shared_ptr<constraints::Constraint<ConType>> con, int k);

  /**
   * @brief 统计第 k 个结点处约束向量的长度
   *
   * 注意：这是每个约束函数输出维度的总和。
   *
   * @param k 结点索引 0 <= k <= N
   * @return 该结点处约束向量的长度
   */
  int NumConstraints(const int k) {
    ALTRO_ASSERT(0 <= k && k <= N_, "k outside valid knot point indices.");
    int cnt = 0;
    for (const constraints::ConstraintPtr<constraints::Equality>& con : eq_.at(k)) {
      cnt += con->OutputDimension();
    }
    for (const constraints::ConstraintPtr<constraints::Inequality>& con : ineq_.at(k)) {
      cnt += con->OutputDimension();
    }
    return cnt;
  }

  /**
   * @brief 统计整个问题中的约束总长度
   *
   * @return 约束总数
   */
  int NumConstraints() {
    int cnt = 0;
    for (int k = 0; k <= N_; ++k) {
      cnt += NumConstraints(k);
    }
    return cnt;
  }

  /**
   * @brief 获取初始状态
   *
   * @return 初始状态向量的引用
   */
  const VectorXd& GetInitialState() const { return *initial_state_; }

  const std::shared_ptr<VectorXd> GetInitialStatePointer() const { return initial_state_; }
  /**
   * @brief 获取第 k 个时间步的代价函数对象
   *
   * @param k 必须在 [0, N] 范围内
   * @return 代价函数对象的共享指针
   */
  std::shared_ptr<CostFunction> GetCostFunction(int k) const {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    return costfuns_[k];
  }

  /**
   * @brief 获取第 k 个时间步的动力学模型对象
   *
   * 若请求最后一个结点，将返回空指针。否则，非法的结点索引会触发断言失败。
   *
   * 若该结点处未定义动力学模型，本函数也会触发断言失败。
   *
   * @param k 必须在 [0, N) 范围内
   * @return 动力学对象的共享指针；若请求最后一个时间步的动力学，则返回 nullptr。
   *
   */
  std::shared_ptr<DiscreteDynamics> GetDynamics(int k) const {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    ALTRO_ASSERT(models_[k] != nullptr, "Dynamics have not been defined at this knot point.");
    return models_[k];
  }

  const std::vector<ConstraintSet<constraints::Equality>>& GetEqualityConstraints() const {
    return eq_;
  }
  const std::vector<ConstraintSet<constraints::Inequality>>& GetInequalityConstraints() const {
    return ineq_;
  }

  int NumSegments() const { return N_; }

  /**
   * @brief 检查问题是否已完整定义
   *
   * 若所有代价与动力学函数指针均非空，且初始状态与首个时间步的状态维度一致，
   * 则认为问题已完整定义。
   *
   * @param verbose 是否输出逐结点检查结果
   * @return true/false 是否完整
   */
  bool IsFullyDefined(bool verbose = false) const;

 private:
  int N_;  // 区段数量（结点数 - 1）
  const std::shared_ptr<VectorXd> initial_state_ = std::make_shared<VectorXd>(0);  // 初始状态
  std::vector<std::shared_ptr<CostFunction>> costfuns_;
  std::vector<std::shared_ptr<DiscreteDynamics>> models_;

  std::vector<ConstraintSet<constraints::Equality>> eq_;
  std::vector<ConstraintSet<constraints::Inequality>> ineq_;
};

}  // namespace problem
}  // namespace altro