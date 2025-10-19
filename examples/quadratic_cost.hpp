// Copyright [2021] Optimus Ride Inc.

#pragma once

#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace examples {

/**
 * 二次型代价函数 QuadraticCost
 *
 * 形式化定义（对单一阶段或终端阶段）：
 *   J(x, u) = 1/2 x^T Q x + x^T H u + 1/2 u^T R u + q^T x + r^T u + c
 *  其中：
 *   - Q ∈ R^{n×n}：状态二次项权重（应对称，半正定）
 *   - R ∈ R^{m×m}：控制二次项权重（应对称，正定，若为非终端阶段）
 *   - H ∈ R^{n×m}：状态-控制耦合项权重
 *   - q ∈ R^{n}   ：状态一次项权重
 *   - r ∈ R^{m}   ：控制一次项权重
 *   - c ∈ R       ：常数项
 *
 * 该类实现了代价的数值计算（Evaluate）、一阶梯度（Gradient）与二阶导（Hessian），
 * 并在构造时对维度、一致性与（半）正定性进行校验（Validate）。
 */
class QuadraticCost : public problem::CostFunction {
 public:
  /**
   * 使用完整参数构造一个二次型代价函数。
   * @param Q 状态二次项权重矩阵（n×n，对称，半正定）
   * @param R 控制二次项权重矩阵（m×m，对称，非终端阶段需正定）
   * @param H 状态-控制耦合矩阵（n×m）
   * @param q 状态线性项（n）
   * @param r 控制线性项（m）
   * @param c 常数项
   * @param terminal 是否为终端阶段代价（true 则放宽对 R 正定性的强制校验）
   */
  QuadraticCost(const MatrixXd& Q, const MatrixXd& R, const MatrixXd& H, const VectorXd& q,
                const VectorXd& r, double c = 0, bool terminal = false)
      : n_(q.size()),
        m_(r.size()),
        isblockdiag_(H.norm() < 1e-8),
        Q_(Q),
        R_(R),
        H_(H),
        q_(q),
        r_(r),
        c_(c),
        terminal_(terminal) {
    Validate();
  }

  /**
   * 便捷构造：以 LQR 形式（关于参考轨迹 xref, uref 的偏差）生成二次代价。
   *
   * 令 e_x = x - xref, e_u = u - uref，则标准 LQR 代价为：
   *   1/2 e_x^T Q e_x + 1/2 e_u^T R e_u
   * 展开后可写成通用二次型：
   *   1/2 x^T Q x + 1/2 u^T R u + q^T x + r^T u + c
   * 其中 q = -Q xref, r = -R uref, c = 1/2 xref^T Q xref + 1/2 uref^T R uref。
   * @param Q 状态权重
   * @param R 控制权重
   * @param xref 状态参考
   * @param uref 控制参考
   * @param terminal 是否为终端代价（影响 R 的校验）
   */
  static QuadraticCost LQRCost(const MatrixXd& Q, const MatrixXd& R, const VectorXd& xref,
                               const VectorXd& uref, bool terminal = false) {
    int n = Q.rows();
    int m = R.rows();
    ALTRO_ASSERT(xref.size() == n, "xref is the wrong size.");
    MatrixXd H = MatrixXd::Zero(n, m);
    VectorXd q = -(Q * xref);
    VectorXd r = -(R * uref);
    double c = 0.5 * xref.dot(Q * xref) + 0.5 * uref.dot(R * uref);
    return QuadraticCost(Q, R, H, q, r, c, terminal);
  }

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }
  /** 计算代价函数值 J(x,u)。*/
  double Evaluate(const VectorXdRef& x,
                  const VectorXdRef& u) override;
  /** 计算一阶梯度：dx = ∂J/∂x, du = ∂J/∂u。*/
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  /** 计算二阶导：dxdx = ∂²J/∂x², dxdu = ∂²J/∂x∂u, dudu = ∂²J/∂u²。*/
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

  /** 获取 Q 矩阵（状态二次项权重）。*/
  const MatrixXd& GetQ() const { return Q_; }
  /** 获取 R 矩阵（控制二次项权重）。*/
  const MatrixXd& GetR() const { return R_; }
  /** 获取 H 矩阵（状态-控制耦合项）。*/
  const MatrixXd& GetH() const { return H_; }
  /** 获取 q 向量（状态线性项）。*/
  const VectorXd& Getq() const { return q_; }
  /** 获取 r 向量（控制线性项）。*/
  const VectorXd& Getr() const { return r_; }
  /** 获取常数项 c。*/
  double GetConstant() const { return c_; }
  /** 获取 Q 的 LDLT 分解结果（用于判定半正定性等）。*/
  const Eigen::LDLT<MatrixXd>& GetQfact() const { return Qfact_; }
  /** 获取 R 的 LLT（Cholesky）分解结果（用于判定正定性等）。*/
  const Eigen::LLT<MatrixXd>& GetRfact() const { return Rfact_; }
  /** 若 H 近似为零矩阵（范数 < 1e-8），认为代价在 (x,u) 上是块对角的。*/
  bool IsBlockDiagonal() const { return isblockdiag_; }

 private:
  /**
   * 校验维度与数值性质：
   * - Q, R, H 的维度与 (n,m) 一致性
   * - Q, R 的对称性（数值上 isApprox 判断）
   * - 非终端阶段：R 必须正定（LLT 成功）
   * - Q 必须半正定（LDLT 成功且主对角元素非负）
   */
  void Validate();

  int n_;
  int m_;
  // 若 H 的范数足够小（近零），则可认为代价无 x-u 耦合。
  bool isblockdiag_;
  // 代价参数：参见类注释中的公式定义。
  MatrixXd Q_;
  MatrixXd R_;
  MatrixXd H_;
  VectorXd q_;
  VectorXd r_;
  double c_;
  // 是否为终端阶段代价：影响对 R 正定性的强制校验与下游使用场景。
  bool terminal_;

  // decompositions of Q and R
  Eigen::LDLT<MatrixXd> Qfact_;
  Eigen::LLT<MatrixXd> Rfact_;
};

}  // namespace examples
}  // namespace altro