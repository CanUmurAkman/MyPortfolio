#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


# In[2]:


# Original distance matrix
distance_matrix = np.array([
    [0, 509, 501, 312, 1019, 736, 656, 60, 1039, 726, 2314, 479, 448, 479, 619, 150],
    [0, 0, 126, 474, 1526, 1226, 1133, 532, 1449, 1122, 2789, 958, 941, 978, 1127, 542],
    [0, 0, 0, 541, 1516, 1184, 1084, 536, 1371, 1045, 2728, 913, 904, 946, 1115, 499],
    [0, 0, 0, 0, 1157, 980, 919, 271, 1333, 1029, 2553, 751, 704, 720, 783, 455],
    [0, 0, 0, 0, 0, 478, 583, 996, 858, 855, 1504, 677, 651, 600, 401, 1033],
    [0, 0, 0, 0, 0, 0, 115, 740, 470, 379, 1581, 271, 289, 261, 308, 687],
    [0, 0, 0, 0, 0, 0, 0, 667, 455, 288, 1661, 177, 216, 207, 343, 592],
    [0, 0, 0, 0, 0, 0, 0, 0, 1066, 759, 2320, 493, 454, 479, 598, 206],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 328, 1387, 591, 650, 656, 776, 933],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1697, 333, 400, 427, 622, 610],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1838, 1868, 1841, 1789, 2248],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 105, 336, 417],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 287, 406],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 237, 449],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 636],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


# In[3]:


# Generate random numbers between 1 and 2
random_numbers = np.random.uniform(1, 2, size=distance_matrix.shape)


# In[4]:


# Display the generated random numbers as a DataFrame
random_numbers_df = pd.DataFrame(random_numbers, index=range(1, 17), columns=range(1, 17))
print("Generated Random Numbers:")
print(random_numbers_df)


# In[5]:


# Multiply the distance matrix by the random numbers
NEW_distance_matrix = distance_matrix * random_numbers


# In[6]:


#Display NEW distance matrix
df_NEW_distance_matrix = pd.DataFrame(NEW_distance_matrix, index=range(1, 17), columns=range(1, 17))
print(df_NEW_distance_matrix)


# In[7]:


# Convert the matrix into a symmetric matrix with diagonal elements zero
symmetric_matrix = np.triu(NEW_distance_matrix) + np.tril(NEW_distance_matrix.T, -1)

# Create a DataFrame
df_symmetric_matrix = pd.DataFrame(symmetric_matrix, index=range(1, 17), columns=range(1, 17))

# Print the DataFrame
print(df_symmetric_matrix)


# In[8]:


# Create a model
model = gp.Model("TSP")


# In[9]:


# Create decision variables
x = model.addVars(range(1, 17), range(1, 17), vtype=GRB.BINARY, name="x")


# In[10]:


# Ensure x[i, j] is 0 when i = j
for i in range(1, 17):
    for j in range(1, 17):
        if i == j:
            model.addConstr(x[i, j] == 0)


# In[11]:


# Ensuring that the graph is an undirected graph [ xij=xji ]
for i in range(1, 17):
    for j in range(1, 17):
        model.addConstr(x[i, j] == x[j, i])


# In[12]:


#Constraints


# In[13]:


# Total Number of Edges Picked = Total Number of Vertices in the Graph * 2 (*2 because one edge counts as two)
model.addConstr(gp.quicksum(x[i, j] for j in range(1, 17) for i in range(1, 17)) == 32)


# In[14]:


# Eliminate cycles of length 2
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            if i != j and j != k and i != k:
                model.addConstr(x[i, j] + x[i, k] + x[j, k] <= 2, f"no_cycle_{i}_{j}_{k}")


# In[15]:


# Eliminate cycles of length 3
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            for l in range(1, 17):
                if i != j and i != k and i != l and j != k and j != l and k != l:
                    model.addConstr(x[i, j] + x[i, k] + x[i, l] + x[j, k] + x[j, l] + x[k, l] <= 3, f"no_cycle_{i}_{j}_{k}_{l}")


# In[16]:


# Eliminate cycles of length 4
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            for l in range(1, 17):
                for m in range(1, 17):
                    if i != j and i != k and i != l and i != m and j != k and j != l and j != m and k != l and k != m and l != m:
                        model.addConstr(x[i, j] + x[i, k] + x[i, l] + x[i, m] + x[j, k] + x[j, l] + x[j, m] + x[k, l] + x[k, m] + x[l, m] <= 4, f"no_cycle_{i}_{j}_{k}_{l}_{m}")


# In[17]:


# Specific Subtour Elimination Constraint 1
model.addConstr(x[14, 16] + x[16, 6] + x[6, 7] + x[7, 12] + x[12, 13] + x[13, 14] <= 5)


# In[18]:


# Specific Subtour Elimination Constraint 2
model.addConstr(x[6, 16] + x[16, 12] + x[12, 13] + x[13, 14] + x[14, 7] + x[7, 6] <= 5)


# In[19]:


# Specific Subtour Elimination Constraint 3
model.addConstr(x[16, 13] + x[13, 14] + x[14, 12] + x[12, 7] + x[7, 5] + x[5, 16] <= 5)


# In[20]:


# Specific Subtour Elimination Constraint 4
model.addConstr(x[6, 16] + x[16, 13] + x[13, 14] + x[14, 12] + x[12, 7] + x[7, 6] <= 5)


# In[21]:


# Ensure each node has exactly two incident edges (2-Matching Constraint, otherwise we need too many constraints)
for i in range(1, 17):
    model.addConstr(gp.quicksum(x[i, j] for j in range(1, 17)) == 2, f"degree_{i}")


# In[24]:


# Set objective: minimize the sum of distances * x[i, j]
model.setObjective(
    gp.quicksum(symmetric_matrix[i-1, j-1] * x[i, j] for i in range(1, 16) for j in range(1, 16)),
    GRB.MINIMIZE
)

# Optimize the model
model.optimize()


# In[25]:


# Print the solution
if model.status == GRB.OPTIMAL:
    for i in range(1, 17):
        for j in range(1, 17):
            if x[i, j].x > 0.5: #0.5 is arbitrary,the aim is to pick x[i,j]=1
                print(f"Edge ({i}, {j}) is selected in the optimal solution.")
else:
    print("No optimal solution found.")

