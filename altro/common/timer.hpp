// 版权声明 [2021] Optimus Ride Inc.

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include <string>

namespace altro {

class Stopwatch;

/**
 * @brief 收集用户指定的多个函数/范围的计时信息。
 * 
 * 本类引入一定开销，因此建议仅用于耗时大于约 100 微秒的代码段
 * （在桌面电脑上创建/销毁一次 Stopwatch 的开销约为 10 微秒）。
 * 
 * 该类只能通过静态方法 MakeShared 或 MakeUnique 以指针形式实例化。
 * 
 * 注意：计时器默认处于非激活状态。调用 Activate() 才会开始采集性能信息。
 * 若计时器未激活，创建一个空的 Stopwatch 对象几乎没有开销，对性能影响可以忽略。
 * 
 * 当需要对某个作用域计时时，调用 Start 生成一个 Stopwatch；当该对象离开作用域时，
 * 会自动将耗时记录到计时器中。每个 Stopwatch 可指定名称以唯一标识被分析的代码，
 * 多次进入相同名称的代码段会在计时器中累加总耗时。
 * 
 * 当计时器被销毁时会打印性能汇总结果，除非之前已调用 PrintSummary() 打印过。
 */
class Timer : public std::enable_shared_from_this<Timer> {
  using microseconds = std::chrono::microseconds; 
 public:
  ~Timer();
  static std::shared_ptr<Timer> MakeShared() {
    return std::shared_ptr<Timer>(new Timer());
  }
  static std::shared_ptr<Timer> MakeUnique() {
    return std::unique_ptr<Timer>(new Timer());
  }
  Stopwatch Start(const std::string& name);
  void PrintSummary();
  void PrintSummary(std::map<std::string, std::chrono::microseconds>* times);
  void Activate() { active_ = true; }
  void Deactivate() { active_ = false; }
  bool IsActive() const { return active_; }
  void SetOutput(FILE* io) { io_ = io; }  // 资源所有权仍由调用者持有
  void SetOutput(const std::string& filename);  // 取得文件资源的所有权
  friend Stopwatch;  // 允许 Stopwatch 访问以写入 times_

 private:
  Timer() = default;  // 构造函数设为私有，仅允许通过智能指针创建
  std::vector<std::string> stack_;  // 当前的调用栈，例如 "al/ilqr/cost"
  std::map<std::string, std::chrono::microseconds> times_;
  bool active_ = false;
  bool printed_summary_ = false;
  bool using_file_ = false;
  FILE* io_ = stdout;
};
using TimerPtr = std::shared_ptr<Timer>;

/**
 * @brief 记录从创建到析构之间的耗时。
 * 
 * 该类仅供 Timer 使用，且只有 Timer 能够调用其构造函数。
 * 当对象析构时，会将本次测量的持续时间自动记录到生成它的父 Timer 中。
 */
class Stopwatch {
  using microseconds = std::chrono::microseconds; 

 public:
  ~Stopwatch();
  friend Stopwatch Timer::Start(const std::string& name); // 允许 Timer 创建 Stopwatch

 protected:
  Stopwatch() = default;
  Stopwatch(std::string name, std::shared_ptr<Timer> timer);

 private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::shared_ptr<Timer> parent_;
};

}  // namespace altro