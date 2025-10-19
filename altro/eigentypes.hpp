// Copyright [2021] Optimus Ride Inc.

#pragma once  // 仅编译一次的头文件保护（比 include guard 更简洁）

#include <eigen3/Eigen/Dense>  // 引入 Eigen 的密集矩阵/向量类型

namespace altro {  // Altro 项目的公共 Eigen 类型别名

// n 维列向量（列数为 1），元素类型默认为 double，可在模板参数中重载
template <int n, class T = double>
using VectorN = Eigen::Matrix<T, n, 1>;

// n 维列向量，元素类型固定为 double（常用别名）
template <int n>
using VectorNd = Eigen::Matrix<double, n, 1>;

// n×m 的矩阵，元素类型为 double（列主存储，Eigen 默认）
template <int n, int m>
using MatrixNxMd = Eigen::Matrix<double, n, m>;

// 对动态大小 double 向量的只读引用视图（零拷贝、遵循原内存布局）
using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>; 

// 行主存储的 n×m double 矩阵（与某些库/缓冲区交互时更方便）
template <int n, int m>
using RowMajorNxMd = Eigen::Matrix<double, n, m, Eigen::RowMajor>;
// 动态大小、行主存储的 double 矩阵
using RowMajorXd = RowMajorNxMd<Eigen::Dynamic, Eigen::Dynamic>;

// 动态大小的列向量（double / float）
using VectorXd = Eigen::VectorXd;
using VectorXf = Eigen::VectorXf;

// 动态大小的矩阵（double / float）
using MatrixXd = Eigen::MatrixXd;
using MatrixXf = Eigen::MatrixXf;

}  // namespace altro 