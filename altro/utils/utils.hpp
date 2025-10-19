// 版权 [2021] Optimus Ride Inc.

#pragma once

#include "altro/utils/assert.hpp"

/**
 * @brief 显式声明某变量未使用
 * 抑制未使用变量的编译警告
 * 
 */
#define ALTRO_UNUSED(var) (void) (var)