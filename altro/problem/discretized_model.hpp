// Copyright [2021] Optimus Ride Inc.

#pragma once

#include "altro/problem/integration.hpp"

namespace altro {
namespace problem {

/**
 * @brief 使用显式积分器将连续动力学模型离散化。
 * 
 * 使用指定的积分器，在一个离散时间步内对连续时间动力学模型进行积分。
 * 
 * @tparam Model 需要被离散化的模型。必须继承自 FunctionBase。
 * @tparam Integrator 显式积分器。应继承自 ExplicitIntegrator。
 * 
 * 为获得最佳性能，应通过 `Model::NStates` 和 `Model::NControls` 提供
 * 状态与控制数量的编译期信息。这将允许积分器在栈上为任何临时数组
 * 分配内存，以满足积分过程中的需要。
 */
template <class Model, class Integrator = RungeKutta4<Model::NStates, Model::NControls>>
class DiscretizedModel : public DiscreteDynamics {
 public:
  static_assert(std::is_base_of<FunctionBase, Model>::value, "Model must inherit from FunctionBase.");
  using DiscreteDynamics::Evaluate;

  static constexpr int NStates = Model::NStates;
  static constexpr int NControls = Model::NControls;

  explicit DiscretizedModel(const Model& model)
      : model_(std::make_shared<Model>(model)), integrator_(model.StateDimension(), model.ControlDimension()) {}

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
                       Eigen::Ref<VectorXd> xnext) override {
    integrator_.Integrate(model_, x, u, t, h, xnext);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
                Eigen::Ref<MatrixXd> jac) override {
    integrator_.Jacobian(model_, x, u, t, h, jac);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
               const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(h);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  bool HasHessian() const override { return model_->HasHessian(); }
  int StateDimension() const override { return model_->StateDimension(); }
  int ControlDimension() const override { return model_->ControlDimension(); }

  Integrator& GetIntegrator() { return integrator_; }

 private:
  std::shared_ptr<Model> model_;
  Integrator integrator_;
};

}  // namespace problem
}  // namespace altro