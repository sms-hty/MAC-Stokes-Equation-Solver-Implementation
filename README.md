# MAC-Stokes-Equation-Solver-Implementation

# #引言以及快速测试指南

本篇文章旨在介绍我的代码中的各个函数, 部分说明可能与报告中的内容相结合。

若您需要快速测试我的代码，复现报告中的结果。我为题目一二三分别准备了两个文件以供测试(一个实时脚本.mlx文件，一个脚本.m文件,
它们中任意一个都能胜任测试任务)，它们分别如下,
您设置好超参后直接运行即可：

1.  problem_1.m,
    problem1.mlx为第一题的测试脚本，里面已经标注了所有超参数并做好了注释。

2.  problem_2.m,
    problem2.mlx为第二题的测试脚本，里面已经标注了所有超参数并做好了注释。

3.  problem_3.m,
    problem3.mlx为第三题的测试脚本，里面已经标注了所有超参数并做好了注释。

注：使用设备为gpu时因为CUDA启动等原因，通常最开始的样例较慢，若需要测时间，推荐重复测试

# #提升限制算子

$\text{lift.m, lift\_UV.m}$为提升算子的函数，前者提升对象为$U, V, W$，后者提升对象仅为$U, V$

$\text{restrict.m, restrict\_UV.m}$为限制算子的函数，前者限制对象为$U, V, W$，后者限制对象仅为$U, V$

# #矩阵作用函数

$\text{apply\_A.m, apply\_B.m, apply\_Btrans.m}$分别为矩阵$\mathbf{A, B, B^T}$的作用函数

# #初始化以及辅助函数

initialize_v\_cycle.m：计算外力矩阵$F_U,F_V$,
测试函数$u, v$在相应位置的值构成的矩阵$U_0, V_0$用于计算误差。

figure_make.m给定数据绘制对数-对数图

# #第一题所使用的专用函数

V_cycle.m: 根据输入参数使用DGS迭代和Vcycle多重网格求解离散Stokes方程

RBGS_iteration.m: 实现了红黑GS迭代以及DGS迭代。

# #第二题所使用的专用函数

Uzawa_iteration.m: 实现Uzawa迭代，其中使用共轭梯度法更新$\mathbf{U}$

conjugate_grad.m: 实现共轭梯度法求解Stokes方程中的速度分量。

# #第三题所使用的专用函数

Inexact_Uzawa_iteration.m：封装了Inexact Uzawa iteration

pre_conjugate_grad.m：封装了Inexact Uzawa iteration中的预优共轭梯度算法

pre_V\_cycle.m：封装了PCG中的Vcycle多重网格以近似预条件解$\hat{U}_k$

sym_GS_iteration.m：实现了顺序对称GS迭代。

sym_RBGS_iteration.m：实现了红黑对称GS迭代，但其收敛效率略低，故未使用(也没有针对优化如只对细网格使用GPU加速)，若您想尝试GPU求解第三问，请将pre_V\_cycle.m中的所有sym_GS_iteration函数替换为sym_RBGS_iteration函数(直接换名字就可以)
