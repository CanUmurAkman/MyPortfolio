#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np


# In[2]:


# Define your data
J = range(1, 6)  # Set of possible server locations
I = range(1, 26)  # Set of clients
S = range(1, 26)  # Set of scenarios


# In[3]:


# Parameters
c = {1: 300, 2: 300, 3: 300, 4: 300, 5: 300}  # Fixed cost of locating a server at location j
u = 592.2  # Capacity of a server
q_j0 = {j: 10000 for j in J}  # Shortage cost at server location j


# In[4]:


print(q_j0)


# In[5]:


# Use d_ij for q_ij
d_ij = {
    (1, 1): 80, (1, 2): 76, (1, 3): 74, (1, 4): 75, (1, 5): 72,
    (2, 1): 79, (2, 2): 77, (2, 3): 71, (2, 4): 71, (2, 5): 73,
    (3, 1): 76, (3, 2): 75, (3, 3): 72, (3, 4): 76, (3, 5): 77,
    (4, 1): 80, (4, 2): 79, (4, 3): 79, (4, 4): 72, (4, 5): 80,
    (5, 1): 74, (5, 2): 75, (5, 3): 70, (5, 4): 71, (5, 5): 80,
    (6, 1): 78, (6, 2): 78, (6, 3): 76, (6, 4): 75, (6, 5): 76,
    (7, 1): 72, (7, 2): 79, (7, 3): 75, (7, 4): 72, (7, 5): 78,
    (8, 1): 70, (8, 2): 71, (8, 3): 73, (8, 4): 76, (8, 5): 78,
    (9, 1): 77, (9, 2): 73, (9, 3): 73, (9, 4): 72, (9, 5): 73,
    (10, 1): 72, (10, 2): 76, (10, 3): 70, (10, 4): 79, (10, 5): 75,
    (11, 1): 77, (11, 2): 78, (11, 3): 72, (11, 4): 76, (11, 5): 79,
    (12, 1): 78, (12, 2): 73, (12, 3): 74, (12, 4): 79, (12, 5): 75,
    (13, 1): 76, (13, 2): 76, (13, 3): 74, (13, 4): 78, (13, 5): 79,
    (14, 1): 74, (14, 2): 79, (14, 3): 79, (14, 4): 80, (14, 5): 78,
    (15, 1): 77, (15, 2): 78, (15, 3): 72, (15, 4): 79, (15, 5): 73,
    (16, 1): 80, (16, 2): 76, (16, 3): 79, (16, 4): 79, (16, 5): 73,
    (17, 1): 70, (17, 2): 75, (17, 3): 71, (17, 4): 77, (17, 5): 75,
    (18, 1): 71, (18, 2): 80, (18, 3): 76, (18, 4): 75, (18, 5): 79,
    (19, 1): 70, (19, 2): 70, (19, 3): 72, (19, 4): 70, (19, 5): 78,
    (20, 1): 75, (20, 2): 73, (20, 3): 79, (20, 4): 74, (20, 5): 72,
    (21, 1): 70, (21, 2): 75, (21, 3): 79, (21, 4): 72, (21, 5): 78,
    (22, 1): 70, (22, 2): 70, (22, 3): 74, (22, 4): 80, (22, 5): 71,
    (23, 1): 70, (23, 2): 73, (23, 3): 80, (23, 4): 70, (23, 5): 72,
    (24, 1): 70, (24, 2): 79, (24, 3): 77, (24, 4): 74, (24, 5): 74,
    (25, 1): 78, (25, 2): 70, (25, 3): 72, (25, 4): 76, (25, 5): 79,
}


# In[6]:


# Client presence data (1st index = i & 2nd index: s) (INDEX STARTS FROM 0)
# Define his as a 25x25 matrix
h_is = np.array([
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
])


# In[7]:


h_is[0,0]


# In[8]:


# Use d_ij for q_ij
q_ij = d_ij


# In[9]:


print(q_ij)


# In[10]:


# Probabilities for each scenario
p_s = {
    1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04,
    6: 0.04, 7: 0.04, 8: 0.04, 9: 0.04, 10: 0.04,
    11: 0.04, 12: 0.04, 13: 0.04, 14: 0.04, 15: 0.04,
    16: 0.04, 17: 0.04, 18: 0.04, 19: 0.04, 20: 0.04,
    21: 0.04, 22: 0.04, 23: 0.04, 24: 0.04, 25: 0.04,
}


# In[11]:


p_s[1]


# In[12]:


# Create a Gurobi model
model = gp.Model("Stochastic_Server_Location")


# In[13]:


# Decision Variables
x = model.addVars(J, vtype=GRB.BINARY, name="x")
y = model.addVars(I, J, S, vtype=GRB.BINARY, name="y")
z = model.addVars(J, S, name="z")


# In[14]:


# Objective Function
model.setObjective(
    gp.quicksum(c[j] * x[j] for j in J) +
    gp.quicksum(p_s[s] * (gp.quicksum(q_j0[j] * z[j, s] for j in J) - gp.quicksum(q_ij[i, j] * y[i, j, s] for i in I for j in J)) for s in S),
    sense=GRB.MINIMIZE
)


# In[15]:


# Constraints


# In[16]:


# Client Assignment Constraints
model.addConstrs(gp.quicksum(y[i, j, s] for j in J) == h_is[i-1][s-1] for i in I for s in S)


# In[17]:


# Demand Satisfaction Constraints
model.addConstrs(gp.quicksum(d_ij[i, j] * y[i, j, s] for i in I) <= u * x[j] + z[j, s] for s in S for j in J)


# In[18]:


# Non-Negativity Constraints
model.addConstrs(z[j, s] >= 0 for j in J for s in S)


# In[19]:


# Optimize the model
model.optimize()


# In[20]:


print(x)


# In[21]:


# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution:")
    for j in J:
        print(f"x[{j}] = {x[j].x}")

    for s in S:
        for i in I:
            for j in J:
                print(f"y[{i}, {j}, {s}] = {y[i, j, s].x}")

    for s in S:
        for j in J:
            print(f"z[{j}, {s}] = {z[j, s].x}")

    print("Total Cost =", model.objVal)

    # Print single scenario costs
    for s in S:
        single_scenario_cost = gp.quicksum(c[j] * x[j].x for j in J) + gp.quicksum(q_j0[j] * z[j, s].x for j in J) - gp.quicksum(q_ij[i, j] * y[i, j, s].x for i in I for j in J)
        print(f"Single Scenario Cost (Scenario {s}) =", single_scenario_cost)
else:
    print("No solution found.")

