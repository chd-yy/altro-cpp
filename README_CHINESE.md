当然可以，以下是你提供的 **AltroCpp** 项目文档的**完整中文翻译版**，我保留了 Markdown 结构、代码块和原有格式，确保你可以直接在文档中使用👇

---

# AltroCpp

一个由 [Optimus Ride, Inc.](https://www.optimusride.com/) 开发的**非线性轨迹优化库**。
该库实现了原始 [开源 ALTRO 求解器](https://github.com/RoboticExplorationLab/Altro.jl) 的 C++ 版本，该求解器最初由 [Robotic Exploration Lab](https://roboticexplorationlab.org/)（隶属斯坦福大学和卡内基梅隆大学）开发，并作为官方的 Julia 软件包开源发布。

有关算法的详细信息，请参阅以下论文与教程：

* [教程](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)
* [原始论文](https://roboticexplorationlab.org/papers/altro-iros.pdf)
* [锥约束 MPC 论文](https://roboticexplorationlab.org/papers/ALTRO_MPC.pdf)
* [带姿态的规划论文](https://roboticexplorationlab.org/papers/planning_with_attitude.pdf)

---

## 许可证（License）

```
Copyright [2021] Optimus Ride Inc.

本程序是自由软件；您可以在 GNU 通用公共许可证（GPL）第 2 版或（您可选择的）任意更高版本的条款下重新分发或修改它。

本程序的发布希望它能够有用，但不提供任何保证；甚至不保证其适销性或适用于特定目的。详情请参阅 GNU 通用公共许可证。

您应该已经随本程序一同收到 GNU 通用公共许可证副本；若没有，请联系：
Free Software Foundation, Inc.
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
```

---

## 从源码构建

### 安装构建依赖

构建依赖包括 `cmake`、`Eigen`、`fmt` 和 `doxygen`。
在基于 Debian 的系统上，可以使用以下命令安装这些依赖：

```bash
sudo apt-get install cmake libeigen3-dev libfmt-dev doxygen
```

---

### 构建

本库使用 **CMake** 构建系统。要构建源码并编译库，请按照以下标准步骤操作：

```bash
cd altro-cpp         # 进入项目根目录
mkdir build          # 创建构建目录
cmake ..             # 运行 CMake 配置步骤
cmake --build .      # 构建所有 CMake 目标
```

如果希望使用其他生成器（如 **Ninja**），可在配置时指定：

```bash
cmake -G Ninja ..
```

---

### 构建选项（Build Options）

构建系统提供以下可选项：

| 选项                               | 描述                                   | 默认值   |
| -------------------------------- | ------------------------------------ | ----- |
| `ALTRO_RUN_CLANG_TIDY`           | 对源码进行静态分析。必须使用 `clang` 编译器。          | `OFF` |
| `ALTRO_BUILD_TESTS`              | 构建测试套件。                              | `ON`  |
| `ALTRO_BUILD_EXAMPLES`           | 构建 `examples` 目录中的示例代码。              | `ON`  |
| `ALTRO_BUILD_BENCHMARKS`         | 构建 `perf` 目录中的性能测试代码。                | `ON`  |
| `ALTRO_BUILD_COVERAGE`           | 对测试套件运行代码覆盖率分析（实验性）。                 | `OFF` |
| `ALTRO_SET_POSITION_INDEPENDENT` | 使用 `-fPIC` 选项将代码编译为位置无关（通常用于与其他库链接）。 | `ON`  |
| `ALTRO_BUILD_SHARED_LIBS`        | 将所有库构建为动态库而非静态库。                     | `OFF` |

这些选项可在配置阶段通过 `-D` 参数指定，例如：

```bash
cmake -D OPTION1=OFF -D OPTION2=ON ..
```

也可以使用 `ccmake` 或 `cmake-gui` 进行交互式修改，例如：

```bash
cmake-gui ..
```

如果已经运行过配置步骤，也可以直接在构建目录重新运行配置：

```bash
cmake-gui .
```

---

### 安装（Installation）

构建系统提供了安装编译后库与头文件的功能。
安装路径由 `CMAKE_INSTALL_PREFIX` 控制，默认值为 `~/.local`。

若要安装（并可选地指定安装路径），请执行以下命令：

```bash
cmake -DCMAKE_INSTALL_PREFIX=~/.local  # 或通过 cmake-gui 指定
cmake --build . --target install       # 构建安装目标
```

安装后，在 `CMAKE_INSTALL_PREFIX` 下会生成以下结构：

```
include/
  altro/
    augmented_lagrangian/
    common/
    ...
lib/
  cmake/
    AltroCpp/
      AltroCppConfig.cmake
      AltroCppConfigVersion.cmake
      AltroCppTargets.cmake
      ...
  libaugmented_lagrangian.a  # 假设构建为静态库
  libcommon.a
  ...
```

---

### 运行单元测试（Unit Tests）

可以通过 **CTest** 轻松运行单元测试。在 `build/` 目录下执行：

```bash
ctest .
```

---

### 本地生成文档（Documentation）

使用以下命令生成 Doxygen 文档：

```bash
cmake --build . --target doxygen
```

生成的主页位于：

```bash
build/docs/html/index.html
```

---

## 使用该库（Using the Library）

使用该库的最简便方式是通过 **CMake** 将其目标导入到你现有的构建系统中。
若该库已在本地安装，可通过 `find_package` 引入：

```cmake
set(AltroCpp_DIR ~/.local)  # 或者是你在 CMAKE_INSTALL_PREFIX 中指定的安装路径
find_library(AltroCpp 0.3 REQUIRED EXACT)
```

`REQUIRED` 和 `EXACT` 参数可以根据需要省略。
详细说明可参考 CMake 文档中 `find_package` 的部分。

库找到后，可以通过 `altro::altro` 目标进行链接：

```cmake
add_executable(my_program
  main.cpp
)
target_link_libraries(my_program
  PRIVATE
  altro::altro
)
```

该库会自动将 `include` 安装目录添加到包含路径中，
因此所有 `#include` 语句应相对于 `altro` 根目录，例如：

```cpp
#include <iostream>
#include <altro/augmented_lagrangian/al_solver.hpp>

int main() {
  const int NumStates = 4;
  const int NumControls = 2;
  int num_segments = 100;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NumStates, NumControls> solver(num_segments);

  return 0;
}
```

---

