// Copyright [2021] Optimus Ride Inc.

#include <iostream>

#include "altro/utils/assert.hpp"

namespace altro {
namespace utils {
  
/**
 * @brief 带信息的断言，允许开发者附带自定义消息
 *
 * 通常通过 ALTRO_ASSERT(expr, msg) 宏调用。
 *
 * 灵感来源于此 StackOverflow 回答：
 * https://stackoverflow.com/questions/3692954/add-custom-messages-in-assert/3692961
 */
void AssertMsg(bool expr, const std::string& msg, const char *expr_str, int line,
               const char *file) {
  if (!expr) {
    std::cerr << "Assert failed:\t" << msg << "\n"
              << "    Evaluated:\t"
              << "'" << expr_str << "'"
              << " in " << file << "::" << line << "" << std::endl;
    abort();
  }
}

} // namespace utils
} // namespace altro