import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# 定义问题（多目标优化）
class PaperHelicopterProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3,  # 自变量个数：旋翼长度、尾部宽度、尾部高度
                         n_obj=2,  # 目标个数：稳定速度、组装成本
                         n_constr=0,  # 无约束条件
                         xl=np.array([0.08, 0.02, 0.05]),  # 自变量下界
                         xu=np.array([0.15, 0.05, 0.106]))  # 自变量上界

    def _evaluate(self, X, out, *args, **kwargs):
        rotor_length = X[:, 0]  # 旋翼长度
        tail_width = X[:, 1]    # 尾部宽度
        tail_height = X[:, 2]   # 尾部高度

        # 计算目标 1：稳定速度（steady_state_velocity）
        # 使用公式 v^2 * r^2 = E1 * r^3 + E2 * (tail_width * tail_height) + E3
        E1 = 5.58  # 假设的常数
        E2 = 0.30
        E3 = 0.006
        steady_state_velocity = np.sqrt(E1 * rotor_length + E2 * tail_width * tail_height / rotor_length**2 + E3 / rotor_length**2)

        # 计算目标 2：组装成本（assembly_cost）
        # 成本是总面积，包含旋翼面积、尾部面积、机身面积
        body_width = 12.0  # 固定机身宽度
        body_height = 4.0  # 固定机身高度
        assembly_cost = (2 * rotor_length * 6) + (tail_width * tail_height) + (body_width * body_height)

        # 输出两个目标值
        out["F"] = np.column_stack([steady_state_velocity, assembly_cost])

# 初始化问题
problem = PaperHelicopterProblem()

# 使用 NSGA-II 算法求解多目标优化问题
algorithm = NSGA2(pop_size=10)  # 种群大小

# 优化
res = minimize(problem,
               algorithm,
               ('n_gen', 200),  # 迭代次数
               seed=1,
               verbose=True)

# 可视化帕累托前沿
Scatter(title="Pareto Front").add(res.F).show()
# plot = Scatter(title="Pareto Front")
# plot.add(res.F)

# # 获取当前图形并设置坐标轴标签
# plt.xlabel("Steady-State Velocity")
# plt.ylabel("Assembly Cost")

# plot.show()

# 提取帕累托前沿的解
pareto_solutions = res.X  # 帕累托前沿对应的自变量值
pareto_objectives = res.F  # 帕累托前沿对应的目标值

# 打印帕累托前沿解
print("Pareto Solutions (Design Variables):")
print(pareto_solutions)
print("\nPareto Objectives (Steady-State Velocity, Assembly Cost):")
print(pareto_objectives)


# 加权决策矩阵法选择配置
weights = np.array([0.7, 0.3])  # 假设我们更关注稳定速度，权重为0.7
weighted_scores = np.dot(pareto_objectives, weights)  # 计算加权得分
best_index = np.argmin(weighted_scores)  # 找到得分最低的解（最优解）

# 输出最优解
best_solution = pareto_solutions[best_index]
best_objectives = pareto_objectives[best_index]
print("\nBest Solution (Design Variables):", best_solution)
print("Best Objectives (Steady-State Velocity, Assembly Cost):", best_objectives)