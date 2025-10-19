// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <memory>

#include "altro/common/state_control_sized.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief 增广拉格朗日代价函数，在现有代价函数基础上添加线性和二次惩罚代价。
 *
 * 它定义了以下形式的代价函数：
 * \f[
 * f(x,u) + \frac{1}{2 \rho} (|| \Pi_{K^*}(\lambda - \rho c(x, u)) ||_2^2 - || \lambda ||_2^2)
 * \f]
 * 其中 \f$ \lambda \in \mathbb{R}^p \f$ 是拉格朗日乘子，
 * \f$ \rho > 0 \in \mathbb{R} \f$ 是惩罚参数，\f$ \Pi_{K^*}(\cdot) \f$ 是
 * 对偶锥 \f$ K^* \f$ 的投影算子。对于等式约束，这简单地是恒等映射。
 *
 * 约束存储为 ConstraintValues，内部存储对偶变量和惩罚参数。
 *
 * @tparam n 编译时状态维度。
 * @tparam m 编译时控制维度。
 */
template <int n, int m>
class ALCost : public problem::CostFunction {
  template <class ConType>
  using ConstraintValueVec =
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>;

 public:
  ALCost(const int state_dim, const int control_dim) : n_(state_dim), m_(control_dim) { Init(); }

  /**
   * @brief 从问题对象构造新的 ALCost 对象
   *
   * 通过组合指定节点索引处的代价函数和约束来生成 ALCost，排除动力学约束。
   *
   * @param prob 约束轨迹优化问题的描述
   * @param k 节点索引。0 <= k <= prob.NumSegments()
   */
  ALCost(const problem::Problem& prob, const int k)
      : n_(prob.GetDynamics(k)->StateDimension()), m_(prob.GetDynamics(k)->ControlDimension()) {
    SetCostFunction(prob.GetCostFunction(k));
    SetEqualityConstraints(prob.GetEqualityConstraints().at(k).begin(),
                           prob.GetEqualityConstraints().at(k).end());
    SetInequalityConstraints(prob.GetInequalityConstraints().at(k).begin(),
                             prob.GetInequalityConstraints().at(k).end());
    Init();
  }

  /***************************** Getters **************************************/

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }
  static constexpr int NStates = n;
  static constexpr int NControls = m;

  ConstraintValueVec<constraints::Equality>& GetEqualityConstraints() { return eq_; }
  ConstraintValueVec<constraints::Inequality>& GetInequalityConstraints() { return ineq_; }

  /**
   * @brief 计算与代价函数关联的约束向量长度。
   *
   * @return int 约束向量长度，包括所有约束类型（等式、不等式、锥约束等）
   */
  int NumConstraints() {
    int p = 0;
    for (size_t i = 0; i < eq_.size(); ++i) {
      p += eq_[i]->OutputDimension();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      p += ineq_[i]->OutputDimension();
    }
    return p;
  }

  std::shared_ptr<problem::CostFunction> GetCostFunction() { return costfun_; }

  /**
   * @brief 为代价函数中的所有约束添加约束信息。
   * 
   * @param coninfo 约束信息向量。当前代价的约束信息被添加到向量末尾。
   */
  void GetConstraintInfo(std::vector<constraints::ConstraintInfo>* coninfo) {
    for (const auto& conval : eq_) {
      coninfo->emplace_back(conval->GetConstraintInfo());
    }
    for (const auto& conval : ineq_) {
      coninfo->emplace_back(conval->GetConstraintInfo());
    }
  }

  /***************************** Setters **************************************/

  /**
   * @brief 分配标称代价函数
   *
   * @param costfun 指向 CostFunction 接口实例的指针。
   */
  void SetCostFunction(const std::shared_ptr<problem::CostFunction>& costfun) {
    ALTRO_ASSERT(costfun != nullptr, "Cost function cannot be a nullptr.");
    costfun_ = costfun;
  }

  // TODO(bjackson): 找到通过 ConType 模板化来统一这些的方法。
  // 非平凡，因为它需要在泛型模板类中特化方法。

  /**
   * @brief 分配等式约束
   *
   * 接受任意迭代器对。当解引用时，迭代器必须返回
   * std::shared_ptr<Constraint<Equality>>。
   *
   * 为每个约束创建 ConstraintValues 类型，其中存储与约束关联的约束值、雅可比矩阵、
   * 对偶变量和惩罚参数。
   *
   * @tparam Iterator 指向等式约束指针的迭代器。必须可复制。
   * @param begin 起始迭代器
   * @param end 终止迭代器。
   */
  template <class Iterator>
  void SetEqualityConstraints(const Iterator& begin, const Iterator& end) {
    eq_.clear();
    CopyToConstraintValues<Iterator, constraints::Equality>(begin, end, &eq_);
    eq_tmp_ = VectorXd::Zero(eq_.size());
  }

  /**
   * @brief 分配不等式约束
   *
   * 接受任意迭代器对。当解引用时，迭代器必须返回
   * std::shared_ptr<Constraint<Inequality>>。
   *
   * 为每个约束创建 ConstraintValues 类型，其中存储与约束关联的约束值、雅可比矩阵、
   * 对偶变量和惩罚参数。
   *
   * @tparam Iterator 指向不等式约束指针的迭代器。必须可复制。
   * @param begin 起始迭代器
   * @param end 终止迭代器。
   */
  template <class Iterator>
  void SetInequalityConstraints(const Iterator& begin, const Iterator& end) {
    ineq_.clear();
    CopyToConstraintValues<Iterator, constraints::Inequality>(begin, end, &ineq_);
    ineq_tmp_ = VectorXd::Zero(ineq_.size());
  }

  /**
   * @brief 为特定约束设置惩罚参数。
   *
   * 适用于该约束的所有元素。
   *
   * @tparam ConType 约束类型
   * @param rho 惩罚参数 (rho > 0)。
   * @param i 约束索引。0 <= i <= NumConstraintFunctions<ConType>()。
   */
  template <class ConType>
  void SetPenalty(const double rho, const int i) {
    ALTRO_ASSERT(0 <= i && i < NumConstraintFunctions<ConType>(),
                 fmt::format("Invalid constraint index. Got {}, expected to be in range [{},{})", i,
                             0, NumConstraintFunctions<ConType>()));
    if (std::is_same<ConType, constraints::Equality>::value) {
      eq_.at(i)->SetPenalty(rho);
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      ineq_.at(i)->SetPenalty(rho);
    }
  }

  /**
   * @brief 为相同类型的所有约束设置相同的惩罚参数
   *
   * @tparam ConType 约束类型
   * @param rho 惩罚参数 (rho > 0)；
   */
  template <class ConType>
  void SetPenalty(const double rho) {
    int num_cons = NumConstraintFunctions<ConType>();
    for (int i = 0; i < num_cons; ++i) {
      SetPenalty<ConType>(rho, i);
    }
  }

  /**
   * @brief 设置惩罚缩放参数。
   *
   * 惩罚缩放是惩罚参数更新的乘性因子。
   *
   * @tparam ConType 约束类型。
   * @param phi 惩罚缩放参数 (phi > 1)。
   * @param i
   */
  template <class ConType>
  void SetPenaltyScaling(const double phi, const int i) {
    ALTRO_ASSERT(0 <= i && i < NumConstraintFunctions<ConType>(),
                 fmt::format("Invalid constraint index. Got {}, expected to be in range [{},{})", i,
                             0, NumConstraintFunctions<ConType>()));
    if (std::is_same<ConType, constraints::Equality>::value) {
      eq_.at(i)->SetPenaltyScaling(phi);
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      ineq_.at(i)->SetPenaltyScaling(phi);
    }
  }

  /**
   * @brief 为相同类型的所有约束设置相同的惩罚缩放参数
   *
   * @tparam ConType 约束类型
   * @param rho 惩罚缩放参数 (phi > 1)。
   */
  template <class ConType>
  void SetPenaltyScaling(const double phi) {
    int num_cons = NumConstraintFunctions<ConType>();
    for (int i = 0; i < num_cons; ++i) {
      SetPenaltyScaling<ConType>(phi, i);
    }
  }

  /**
   * @brief 获取特定类型约束函数的数量
   *
   * NumConstraintFunctions() <=  NumConstraints() （仅当每个约束函数的输出维度为 1 时相等）。
   *
   * @tparam ConType 约束函数类型（例如 Equality、Inequality 等）
   * @return int 约束函数数量。
   */
  template <class ConType>
  int NumConstraintFunctions() {
    int size = 0.0;
    if (std::is_same<ConType, constraints::Equality>::value) {
      size = eq_.size();
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      size = ineq_.size();
    }
    return size;
  }

  /***************************** Methods **************************************/

  /**
   * @brief 评估增广拉格朗日代价
   *
   * @param x 状态向量
   * @param u 控制向量
   * @return double 标称代价加上约束惩罚的额外项。
   *
   * @pre 在调用此函数之前必须设置代价函数。
   */
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    double J = costfun_->Evaluate(x, u);
    for (size_t i = 0; i < eq_.size(); ++i) {
      J += eq_[i]->AugLag(x, u);
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      J += ineq_[i]->AugLag(x, u);
    }
    return J;
  }

  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    costfun_->Gradient(x, u, dx, du);
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->AugLagGradient(x, u, dx_tmp_, du_tmp_);
      dx += dx_tmp_;
      du += du_tmp_;
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->AugLagGradient(x, u, dx_tmp_, du_tmp_);
      dx += dx_tmp_;
      du += du_tmp_;
    }
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
               Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    costfun_->Hessian(x, u, dxdx, dxdu, dudu);
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->AugLagHessian(x, u, dxdx_tmp_, dxdu_tmp_, dudu_tmp_, full_newton_);
      dxdx += dxdx_tmp_;
      dxdu += dxdu_tmp_;
      dudu += dudu_tmp_;
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->AugLagHessian(x, u, dxdx_tmp_, dxdu_tmp_, dudu_tmp_, full_newton_);
      dxdx += dxdx_tmp_;
      dxdu += dxdu_tmp_;
      dudu += dudu_tmp_;
    }
  }

  /**
   * @brief 对所有约束应用对偶更新
   *
   */
  void UpdateDuals() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->UpdateDuals();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->UpdateDuals();
    }
  }

  /**
   * @brief 对所有约束应用惩罚更新
   *
   */
  void UpdatePenalties() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->UpdatePenalties();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->UpdatePenalties();
    }
  }

  /**
   * @brief 查找当前节点的最大约束违反
   *
   * @tparam p 计算违反时使用的范数（默认为 Infinity）
   * @return 最大约束违反
   */
  template <int p = Eigen::Infinity>
  double MaxViolation() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_tmp_(i) = eq_[i]->template MaxViolation<p>();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_tmp_(i) = ineq_[i]->template MaxViolation<p>();
    }
    Eigen::Vector2d tmp(eq_tmp_.template lpNorm<p>(), ineq_tmp_.template lpNorm<p>());
    return tmp.template lpNorm<p>();
  }

  /**
   * @brief 查找当前节点任何约束使用的最大惩罚参数。
   *
   * @return 所有约束中的最大惩罚参数。
   */
  double MaxPenalty() {
    double max_penalty = 0.0;
    for (size_t i = 0; i < eq_.size(); ++i) {
      max_penalty = std::max(max_penalty, eq_[i]->MaxPenalty());
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      max_penalty = std::max(max_penalty, ineq_[i]->MaxPenalty());
    }
    return max_penalty;
  }

  void ResetDualVariables() {
    for (auto& con: eq_) {
      con->ResetDualVariables();
    }
    for (auto& con: ineq_) {
      con->ResetDualVariables();
    }
  }

 private:
  /**
   * @brief 为任意约束分配新的 ConstraintValue，将指针存储在适当类型的向量中。
   *
   *
   * @tparam Iterator 指向约束指针的任意迭代器。必须可复制。
   * @tparam ConType 约束类型（Equality、Inequality 等）
   * @param begin 起始迭代器。
   * @param end 终止迭代器。
   * @param convals 用于存储新 ConstraintValue 的容器。
   */
  template <class Iterator, class ConType>
  void CopyToConstraintValues(
      const Iterator& begin, const Iterator& end,
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>* convals) {
    ALTRO_ASSERT(convals != nullptr, "Must provide a pointer to a valid collection.");
    for (Iterator it = begin; it != end; ++it) {
      convals->emplace_back(
          std::make_shared<constraints::ConstraintValues<n, m, ConType>>(this->n_, this->m_, *it));
    }
  }

  /**
   * @brief 分配临时存储数组。
   *
   */
  void Init() {
    dx_tmp_.setZero(this->n_);
    du_tmp_.setZero(this->m_);
    dxdx_tmp_.setZero(this->n_, this->n_);
    dxdu_tmp_.setZero(this->n_, this->m_);
    dudu_tmp_.setZero(this->m_, this->m_);
  }

  std::shared_ptr<problem::CostFunction> costfun_;

  // 约束
  ConstraintValueVec<constraints::Equality> eq_;
  ConstraintValueVec<constraints::Inequality> ineq_;

  const int n_;
  const int m_;

  // 使用完整/高斯牛顿的标志
  // TODO(bjackson): 添加选项来更改此设置，并在展开中使用它。
  bool full_newton_ = false;

  VectorXd eq_tmp_;
  VectorXd ineq_tmp_;

  // 用于在添加之前收集代价展开的数组
  // 必须是可变的，因为 CostFunction 接口需要 const 方法
  VectorNd<n> dx_tmp_;
  VectorNd<m> du_tmp_;
  MatrixNxMd<n, n> dxdx_tmp_;
  MatrixNxMd<n, m> dxdu_tmp_;
  MatrixNxMd<m, m> dudu_tmp_;
};

}  // namespace augmented_lagrangian
}  // namespace altro