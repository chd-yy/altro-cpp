// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <limits>
#include <vector>
#include <map>
#include <unordered_map>

#include "altro/utils/assert.hpp"
#include "altro/common/log_entry.hpp"

namespace altro {

/**
 * @brief 提供类似表格的日志输出
 *
 * 日志记录器包含几个不同的数据条目（或字段/列），可以在调用打印函数之间填充。
 * 详细程度可以在运行时修改。它还支持基于数值数据条目简单边界的条件格式化。
 *
 * 每个条目的键是由 LogEntry 指定的标题（在标题中打印的名称）。
 * 
 * # 示例
 * 以下示例创建一个带有整数和浮点条目的简单日志记录器，设置标题颜色、
 * 标题打印频率，并记录和打印一些数据。
 * @code {.cpp}
 * // 创建日志记录器并添加条目
   SolverLogger logger;
   logger.AddEntry(0, "iters", "{:>4}", LogEntry::kInt).SetWidth(6).SetLevel(LogLevel::kOuterDebug);
   logger.AddEntry(1, "cost", "{:.4g}").SetLevel(LogLevel::kOuter);

   // 设置选项
   logger.SetHeaderColor(fmt::color::cyan);
   logger.SetFrequency(5);
   logger.SetLevel(LogLevel::kInner);

   // 记录数据
   logger.Log("iters", 1);
   logger.Log("cost", 10.0);
   logger.Print();
   logger.Log("iters", 2);
   logger.Print();  // 在 "cost" 列中保持 "10"
   logger.Clear() 
 * @endcode
 */
class SolverLogger {
 public:
  /**
   * @brief 构造新的求解器日志记录器对象
   *
   * @param level 详细程度级别。级别为 0 时不打印任何内容。
   */
  explicit SolverLogger(const LogLevel level = LogLevel::kSilent) : cur_level_(level) {}

  LogLevel GetLevel() const { return cur_level_; }
  LogEntry& GetEntry(const std::string& title) { return entries_[title]; }
  int NumEntries() { return entries_.size(); }

  /*************************** Iteration **************************************/
  using iterator = std::unordered_map<std::string, LogEntry>::iterator;
  using const_iterator = std::unordered_map<std::string, LogEntry>::const_iterator;
  iterator begin() { return entries_.begin(); }
  const_iterator begin() const { return entries_.cbegin(); }
  iterator end() { return entries_.end(); }
  const_iterator end() const { return entries_.cend(); }

  /**
   * @brief 向日志记录器添加数据条目/字段/列。
   *
   * @tparam Args
   * @param col 数据列。指定数据应打印的列。如果 col >= 0，它是基于 0 的列索引。
   * 如果 col < 0，它从末尾向后计数，col = -1 将其添加为最后一列。
   * @param args 传递给 LogEntry 构造函数的参数。
   */
  template <class... Args>
  LogEntry& AddEntry(const int& col, Args... args);

  /**
   * @brief 设置日志记录器的详细程度级别。
   * 
   * @param level 
   */
  void SetLevel(const LogLevel level) { cur_level_ = level; }

  /**
   * @brief 禁用所有输出。
   * 
   */
  void Disable() { SetLevel(LogLevel::kSilent); }

  /**
   * @brief 设置打印标题的频率。
   * 
   * 如果频率设置为 5，标题将每 5 次迭代打印一次。
   * 
   * @param freq 
   */
  void SetFrequency(const int freq) {
    ALTRO_ASSERT(freq >= 0, "Header print frequency must be positive.");
    frequency_ = freq;
  }

  /**
   * @brief 记录给定字段的数据。
   *
   * 用户有责任确保提供的数据与给定字段的格式规范一致。
   * 
   * 如果条目在当前详细程度级别下不活跃，则不会记录数据。
   *
   * @tparam T 给定数据的数据类型。
   * @param title 要添加数据的数据列标题。它必须是现有的数据列（但可以是不活跃的）。
   * @param value 要记录、格式化并稍后打印的值。
   */
  template <class T>
  void Log(const std::string& title, T value);

  /**
   * @brief 打印标题
   * 
   * 打印所有活跃条目的标题，后跟水平线和换行符。
   * 如果当前详细程度级别为 0，则不会打印任何内容。
   */
  void PrintHeader();

  /**
   * @brief 打印数据行
   * 
   * 打印所有活跃条目的数据（包括条件格式化）。
   * 如果当前详细程度级别为 0，则不会打印任何内容。
   */
  void PrintData();

  /**
   * @brief 以指定频率自动打印标题
   * 
   */
  void Print();

  /**
   * @brief 清除表中的所有数据条目。
   * 
   */
  void Clear();

  /**
   * @brief 设置标题及其水平线的颜色
   * 
   * @param color fmt 库提供的颜色之一
   * （例如 fmt::color::green、fmt::color::yellow、fmt::color::red、fmt::color::white 等）
   */
  void SetHeaderColor(const fmt::color color) { header_color_ = color; }

 private:
  static constexpr int kDefaultFrequency = 10;

  LogLevel cur_level_ = LogLevel::kSilent;   // 当前详细程度级别
  int frequency_ = kDefaultFrequency;        // 标题打印频率
  int count_ = 0;                            // 自标题以来的打印次数
  std::unordered_map<std::string, LogEntry> entries_;
  std::vector<const std::string*> order_;
  fmt::color header_color_ = fmt::color::white;
};

template <class... Args>
LogEntry& SolverLogger::AddEntry(const int& col, Args... args) {
  ALTRO_ASSERT(
      col <= static_cast<int>(entries_.size()),
      fmt::format("Column ({}) must be less than or equal to the current number of entries ({}).",
                  col, NumEntries()));
  ALTRO_ASSERT(
      col >= -static_cast<int>(entries_.size()) - 1,
      fmt::format(
          "Column ({}) must be greater or equal to than the negative new number of entries ({}).",
          col, -NumEntries() - 1));

  // Create a LogEntry and forward it to the map after extracting the title
  LogEntry entry(std::forward<Args>(args)...);
  const std::string title = entry.GetTitle();
  auto insert = entries_.emplace(std::make_pair(title, std::move(entry)));

  // Get a pointer to the title string used in the map
  // insert.first is an iterator over key-value pairs for the map entries_
  const std::string* title_ptr = &(insert.first->first);

  // Specify the output order
  auto it = order_.begin();
  if (col < 0) {
    // Count from the back if the input is negative
    it = order_.end() + col + 1;
  } else {
    it += col;
  }
  order_.insert(it, title_ptr);

  return insert.first->second;
}

template <class T>
void SolverLogger::Log(const std::string& title, T value) {
  // Short-circuit to skip the hash lookup if logging is disabled
  if (cur_level_ > LogLevel::kSilent && entries_[title].IsActive(cur_level_)) {
    entries_[title].Log(value);
  }
}


}  // namespace altro
