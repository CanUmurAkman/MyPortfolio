#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np


# In[2]:


# Define your data
J_range = range(0, 5)  # Set of possible server locations
I_range = range(0, 25)  # Set of clients
S_range = range(0, 25)  # Set of scenarios


# In[3]:


# Define constants
J = 5
I = 25
S = 25


# In[4]:


# Parameters
c = [300] * 5  # Fixed cost of locating a server at location j
u = 592.2  # Capacity of a server
q_j0 = {j: 10000 for j in J_range}  # Shortage cost at server location j


# In[5]:


c[0]


# In[6]:


# Define dij as a 25x5 matrix
d_ij = np.array([
    [80, 76, 74, 75, 72],
    [79, 77, 71, 71, 73],
    [76, 75, 72, 76, 77],
    [80, 79, 79, 72, 80],
    [74, 75, 70, 71, 80],
    [78, 78, 76, 75, 76],
    [72, 79, 75, 72, 78],
    [70, 71, 73, 76, 78],
    [77, 73, 73, 72, 73],
    [72, 76, 70, 79, 75],
    [77, 78, 72, 76, 79],
    [78, 73, 74, 79, 75],
    [76, 76, 74, 78, 79],
    [74, 79, 79, 80, 78],
    [77, 78, 72, 79, 73],
    [80, 76, 79, 79, 73],
    [70, 75, 71, 77, 75],
    [71, 80, 76, 75, 79],
    [70, 70, 72, 70, 78],
    [75, 73, 79, 74, 72],
    [70, 75, 79, 72, 78],
    [70, 70, 74, 80, 71],
    [70, 73, 80, 70, 72],
    [70, 79, 77, 74, 74],
    [78, 70, 72, 76, 79],
])


# In[7]:


d_ij[0,0]


# In[8]:


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


# In[9]:


h_is[0,0]


# In[10]:


# Use d_ij for q_ij
q_ij = np.array([
    [80, 76, 74, 75, 72],
    [79, 77, 71, 71, 73],
    [76, 75, 72, 76, 77],
    [80, 79, 79, 72, 80],
    [74, 75, 70, 71, 80],
    [78, 78, 76, 75, 76],
    [72, 79, 75, 72, 78],
    [70, 71, 73, 76, 78],
    [77, 73, 73, 72, 73],
    [72, 76, 70, 79, 75],
    [77, 78, 72, 76, 79],
    [78, 73, 74, 79, 75],
    [76, 76, 74, 78, 79],
    [74, 79, 79, 80, 78],
    [77, 78, 72, 79, 73],
    [80, 76, 79, 79, 73],
    [70, 75, 71, 77, 75],
    [71, 80, 76, 75, 79],
    [70, 70, 72, 70, 78],
    [75, 73, 79, 74, 72],
    [70, 75, 79, 72, 78],
    [70, 70, 74, 80, 71],
    [70, 73, 80, 70, 72],
    [70, 79, 77, 74, 74],
    [78, 70, 72, 76, 79],
])


# In[11]:


# Probabilities for each scenario
p_s = [0.04] * 25


# In[12]:


p_s[0]


# In[13]:


# Create a Gurobi model
model = gp.Model("Expected_Disutility")


# In[14]:


# Decision Variables
x = model.addVars(J_range, vtype=GRB.BINARY, name="x")
y = model.addVars(I_range, J_range, S_range, vtype=GRB.BINARY, name="y")
z = model.addVars(J_range, S_range, name="z")
v_pos = model.addVars(S_range, vtype=GRB.CONTINUOUS, name="v_pos")
v_neg = model.addVars(S_range, vtype=GRB.CONTINUOUS, name="v_neg")


# In[15]:


#Define m
m=5


# In[16]:


# Constraints


# In[17]:


# Non-negativity constraint for v_pos 
model.addConstrs(v_pos[s] >= 0 for s in S_range)

# Non-positivity constraint for v_neg
model.addConstrs(v_neg[s] <= 0 for s in S_range)

#At least one of them must be 0 (otherwise there is a change that they are both non-zero)
model.addConstrs(v_pos[s] * v_neg[s] == 0 for s in S_range)


# In[18]:


#Objective Function Value (v) Constraint
model.addConstrs((v_pos[s] + v_neg[s] == gp.quicksum(c[j] * x[j] for j in J_range) +
    gp.quicksum(q_j0[j] * z[j, s] for j in J_range) - gp.quicksum(q_ij[i, j] * y[i, j, s] for i in I_range for j in J_range)) 
                 for s in S_range)


# In[19]:


# Client Assignment Constraints (OLD)
model.addConstrs(gp.quicksum(y[i, j, s] for j in J_range) == h_is[i, s] for i in I_range for s in S_range)


# In[20]:


# Demand Satisfaction Constraints (OLD)
model.addConstrs(gp.quicksum(d_ij[i, j] * y[i, j, s] for i in I_range) <= u * x[j] + z[j, s] for s in S_range for j in J_range)


# In[21]:


# Non-Negativity Constraints (OLD)
model.addConstrs(z[j, s] >= 0 for j in J_range for s in S_range)


# In[22]:


# Expected Disutility Objective Function
# Set the objective function
model.setObjective(
    gp.quicksum(p_s[s]* m * v_pos[s] for s in S_range) + gp.quicksum(p_s[s]* 1 * v_neg[s] for s in S_range), 
    sense=GRB.MINIMIZE)


# In[23]:


# Optimize the model
model.optimize()


# In[24]:


# Display the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for var in model.getVars():
        print(f"{var.varName}= {var.x}")
    print(f"Optimal Objective Value: {model.objVal}")
else:
    print("No optimal solution found.")


# In[25]:


# Print single scenario costs
for s in S_range:
        single_scenario_cost = gp.quicksum(c[j] * x[j].x for j in J_range) + gp.quicksum(q_j0[j] * z[j, s].x for j in J_range) - gp.quicksum(q_ij[i, j] * y[i, j, s].x for i in I_range for j in J_range)
        print(f"Single Scenario Cost (Scenario {s}) =", single_scenario_cost)

