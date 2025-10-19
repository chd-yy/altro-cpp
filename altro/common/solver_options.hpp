// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <thread>
#include <cmath>

#include "altro/utils/utils.hpp"
#include "altro/common/solver_logger.hpp"

namespace altro {

constexpr int kPickHardwareThreads = -1;

/**
 * @brief 增广拉格朗日和 iLQR 求解器的选项
 * 
 */
struct SolverOptions {
  SolverOptions();
  // clang-format off
  // NOLINT comments added to supress clang-tidy [readibility-magic-numbers] check
  int max_iterations_total = 300;         // NOLINT 总迭代 LQR 迭代的最大次数
  int max_iterations_outer = 30;          // NOLINT 增广拉格朗日迭代的最大次数
  int max_iterations_inner = 100;         // NOLINT 单次求解中的最大 iLQR 迭代次数
  double cost_tolerance = 1e-4;           // NOLINT 代价下降的阈值
  double gradient_tolerance = 1e-2;       // NOLINT 近似梯度无穷范数的阈值

  double bp_reg_increase_factor = 1.6;    // NOLINT 增加正则化的乘法因子
  bool bp_reg_enable = true;              // NOLINT 在后向传递中启用正则化
  double bp_reg_initial = 0.0;            // NOLINT 初始正则化
  double bp_reg_max = 1e8;                // NOLINT 最大正则化
  double bp_reg_min = 1e-8;               // NOLINT 最小正则化
  // double bp_reg_forwardpass = 10.0;     
  int bp_reg_fail_threshold = 100;        // NOLINT 后向传递在抛出错误前可以失败的次数
  bool check_forwardpass_bounds = true;   // NOLINT 是否检查展开是否保持在指定边界内
  double state_max = 1e8;                 // NOLINT 最大状态值（绝对值）
  double control_max = 1e8;               // NOLINT 最大控制值（绝对值）

  int line_search_max_iterations = 20;    // NOLINT 在增加正则化之前的最大线搜索迭代次数
  double line_search_lower_bound = 1e-8;  // NOLINT 充分改进条件
  double line_search_upper_bound = 10.0;  // NOLINT 不能比预期改进太多
  double line_search_decrease_factor = 2; // NOLINT 每次迭代线搜索步长减少多少

  double constraint_tolerance = 1e-4;     // NOLINT 最大约束违反阈值
  double maximum_penalty = 1e8;           // NOLINT 允许的最大惩罚参数
  double initial_penalty = 1.0;           // NOLINT 所有约束的初始惩罚。每次求解前总是将所有惩罚重置为此值。设置为 0 以禁用。
  bool reset_duals = true;                // NOLINT 每次求解前重置对偶变量
  int header_frequency = 10;              // NOLINT AL 迭代的标题打印频率（对于级别 < kInner）
  LogLevel verbose = LogLevel::kSilent;   // 输出详细程度级别
  bool profiler_enable = false;                  // 启用内部分析器
  bool profiler_output_to_file = false;    // 输出到文件（true）或标准输出（false）
  std::string log_directory;
  std::string profile_filename = "profiler.out";
  int nthreads = 1;                        // 要使用的处理器数量。设置为 kPickHardwareThreads 以自动选择。
  int tasks_per_thread = 1;
  // clang-format on

  int NumThreads() const {
    if (nthreads == kPickHardwareThreads) {
      return std::thread::hardware_concurrency();
    }
    return std::max(nthreads, 1);
  }
};

}  // namespace altro