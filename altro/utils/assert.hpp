// Copyright [2021] Optimus Ride Inc.

#pragma once
#include <string>

// 在未定义 NDEBUG（调试模式）时启用断言；定义 NDEBUG（发布模式）时禁用断言
#ifndef NDEBUG
#define ALTRO_ASSERT(Expr, Msg) altro::utils::AssertMsg((Expr), Msg, #Expr, __LINE__, __FILE__)
#else
#define ALTRO_ASSERT(Expr, Msg) ;
#endif

namespace altro {
namespace utils {

// 带消息的断言实现（通常不直接调用，使用 ALTRO_ASSERT 宏）
void AssertMsg(bool expr, const std::string& msg, const char* expr_str, int line, const char* file);

// 在当前编译配置下断言是否生效（编译期常量）
constexpr bool AssertionsActive() { 
#ifndef NDEBUG
  return true; 
#else
  return false;
#endif
}

}  // namespace utils
}  // namespace altro