import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# Define the question（Multi-objective optimization）
class PaperHelicopterProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3,  # Number of independent variables: rotor length, tail width, tail height
                         n_obj=2,  # Target quantity: Stable speed, assembly cost
                         n_constr=0,  # No constraints
                         xl=np.array([0.08, 0.02, 0.05]),  # Lower bound of independent variable
                         xu=np.array([0.15, 0.05, 0.106]))  # Upper bound of independent variable

    def _evaluate(self, X, out, *args, **kwargs):
        rotor_length = X[:, 0]  # Rotor length
        tail_width = X[:, 1]    # Tail width
        tail_height = X[:, 2]   # Tail height

        # Calculate the target 1：steady speed（steady_state_velocity）
        # Use the formula v^2 * r^2 = E1 * r^3 + E2 * (tail_width * tail_height) + E3
        E1 = 5.58  # Assumed constant
        E2 = 0.30
        E3 = 0.006
        steady_state_velocity = np.sqrt(E1 * rotor_length + E2 * tail_width * tail_height / rotor_length**2 + E3 / rotor_length**2)

        # Calculate the target 2：assembly cost（assembly_cost）
        # The cost is the total area, including rotor area, tail area, and fuselage area
        body_width = 12.0  # Fixed body width
        body_height = 4.0  # Fixed body height
        assembly_cost = (2 * rotor_length * 6) + (tail_width * tail_height) + (body_width * body_height)

        # Output two target values
        out["F"] = np.column_stack([steady_state_velocity, assembly_cost])

# Initialize the problem
problem = PaperHelicopterProblem()

# Use NSGA-II algorithm to solve multi-objective optimization problem
algorithm = NSGA2(pop_size=10)  # Population size

# Optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 200),  # Number of iterations
               seed=1,
               verbose=True)

# Visualize Pareto front
Scatter(title="Pareto Front").add(res.F).show()
# plot = Scatter(title="Pareto Front")
# plot.add(res.F)

# Retrieve the current graphic and set axis labels
# plt.xlabel("Steady-State Velocity")
# plt.ylabel("Assembly Cost")

# plot.show()

# Extract the solution of Pareto front
pareto_solutions = res.X  # The independent variable values corresponding to the Pareto front
pareto_objectives = res.F  # The target value corresponding to the Pareto front

# Print Pareto frontier solutions
print("Pareto Solutions (Design Variables):")
print(pareto_solutions)
print("\nPareto Objectives (Steady-State Velocity, Assembly Cost):")
print(pareto_objectives)


# Weighted decision matrix method for selecting configurations
weights = np.array([0.7, 0.3])  # Assuming we focus more on stable speed, with a weight of 0.7
weighted_scores = np.dot(pareto_objectives, weights)  # Calculate weighted score
best_index = np.argmin(weighted_scores)  # Find the solution with the lowest score (optimal solution)

# Output the optimal solution
best_solution = pareto_solutions[best_index]
best_objectives = pareto_objectives[best_index]
print("\nBest Solution (Design Variables):", best_solution)
print("Best Objectives (Steady-State Velocity, Assembly Cost):", best_objectives)
