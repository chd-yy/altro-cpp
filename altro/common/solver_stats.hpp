// Copyright [2021] Optimus Ride Inc.

#pragma once

#include "altro/common/solver_options.hpp"
#include "altro/common/solver_logger.hpp"
#include "altro/common/timer.hpp"
#include "altro/utils/utils.hpp"

namespace altro {



/**
 * @brief 描述求解器的当前状态
 *
 * 用于描述求解器是否成功求解问题或提供失败的原因。
 */
enum class SolverStatus {
  kSolved = 0,
  kUnsolved = 1,
  kStateLimit = 2,
  kControlLimit = 3,
  kCostIncrease = 4,
  kMaxIterations = 5,
  kMaxOuterIterations = 6,
  kMaxInnerIterations = 7,
  kMaxPenalty = 8,
  kBackwardPassRegularizationFailed = 9,
};

/**
 * @brief 保存求解过程中记录的统计信息
 * 
 * 此类还提供以不同详细程度级别向终端输出数据的功能。
 * 
 * 要通过日志记录器记录和打印的任何新数据字段都应该在 DefaultLogger() 中
 * "注册"到日志记录器。此结构中的所有数据字段都不需要"注册"到日志记录器。
 * 
 */
class SolverStats {
 public:
  SolverStats() : timer_(Timer::MakeShared()) { 
    DefaultLogger(); 
  }

  double initial_cost = 0.0;
  int iterations_inner = 0;
  int iterations_outer = 0;
  int iterations_total = 0;
  std::vector<double> cost;
  std::vector<double> alpha;
  std::vector<double> improvement_ratio;  // 实际与预期代价下降的比率
  std::vector<double> gradient;
  std::vector<double> cost_decrease;
  std::vector<double> regularization;
  std::vector<double> violations;     // 每次 AL 迭代的最大约束违反
  std::vector<double> max_penalty;    // 每次 AL 迭代的最大惩罚参数

  /**
   * @brief 设置输出的代价、约束和梯度容差。
   * 
   * 低于这些容差的任何记录值将以绿色打印。
   * 
   * @param cost 代价容差，或迭代间代价的变化。
   * @param viol 最大约束违反。
   * @param grad 梯度的最大范数。
   */
  void SetTolerances(const double& cost, const double& viol, const double& grad);

  /**
   * @brief 设置内部存储向量的容量
   * 
   * @param n 要分配的大小，通常等于最大迭代次数。
   */
  void SetCapacity(int n);

  /**
   * @brief 重置统计信息，清除所有向量并将所有计数器重置为零。
   * 
   */
  void Reset();

  /**
   * @brief 设置控制台日志记录器的详细程度级别
   * 
   * @param level 
   */
  void SetVerbosity(const LogLevel level) { logger_.SetLevel(level); }

  /**
   * @brief 获取控制台日志记录器的详细程度
   * 
   * @return 当前详细程度级别
   */
  LogLevel GetVerbosity() const { return logger_.GetLevel(); }
  SolverLogger& GetLogger() { return logger_; }
  const SolverLogger& GetLogger() const { return logger_; }
  TimerPtr& GetTimer() { return timer_; }
  const TimerPtr& GetTimer() const { return timer_; }
  SolverOptions& GetOptions() { return opts_; }
  const SolverOptions& GetOptions() const { return opts_; }
  std::string ProfileOutputFile();

  /**
   * @brief 将最后一次迭代打印到控制台
   * 
   */
  void PrintLast() { logger_.Print(); }

  /**
   * @brief 记录数据
   * 
   * 此命令执行 2 件事：
   * 1) 它尝试将值发送到日志记录器，在那里它将被格式化为字符串并存储以供稍后打印。
   * 2) 它将值存储在相应的存储向量中，总是保存到向量的最后一个元素。
   * 
   * 如果在调用 NewIteration() 之间多次调用此函数，它将覆盖先前的值。
   * 
   * @tparam T 要记录的值的数据类型。应与数据字段一致。
   * @param title 要记录的值的标题。这与控制台中打印的标题相同。
   * @param value 要记录的值。
   */
  template <class T>
  void Log(const std::string& title, T value) {
    logger_.Log(title, value);
    SetData(title, value);
  }

  /**
   * @brief 将数据向前推进一次迭代，有效地保存所有当前数据。
   * 
   */
  void NewIteration();

  //  TODO(bjackson): Make this private to not confuse the user
  // Requires a friend relationship with SolverOptions
  void ProfilerOutputToFile(bool flag);

 private:

  /**
   * @brief 将数据保存在相应的向量中。
   * 
   * @tparam T 要记录的值的数据类型。
   * @param title 日志条目的标题。与控制台中打印的标题对应。
   * @param value 要保存的值。
   */
  template <class T>
  void SetData(const std::string& title, T value);

  /**
   * @brief 在 floats_ 中创建一个条目，将标题键映射到公开可访问的向量之一。
   * 
   * 在设置 SolverStats 对象时调用此方法。
   * 
   * @tparam T 条目字段的数据类型（double 或 int）。
   * @param entry 数据向量要关联的条目字段。
   * @param data 存储在此类中的公开可访问向量之一。
   */
  template <class T>
  void SetPtr(const LogEntry& entry, std::vector<T>& data) {
    ALTRO_UNUSED(entry);
    ALTRO_UNUSED(data);
  }

  /**
   * @brief 创建默认日志记录字段
   * 
   */
  void DefaultLogger();

  std::unordered_map<std::string, std::vector<double>*> floats_;

  int len_ = 0;
  SolverLogger logger_;
  TimerPtr timer_;
  SolverOptions opts_;
};

template <class T>
void SolverStats::SetData(const std::string& title, T value) {
  // Automatically register the first iteration to prevent accessing empty vectors
  if (len_ == 0) {
    NewIteration();
  }
  // Search for the title
  auto search_float = floats_.find(title);
  if (search_float != floats_.end()) {
    search_float->second->back() = value;
  }
}

}  // namespace altro