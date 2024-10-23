#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

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

# Generate random numbers between 1 and 2
random_numbers = np.random.uniform(1, 2, size=distance_matrix.shape)

# Display the generated random numbers as a DataFrame
random_numbers_df = pd.DataFrame(random_numbers, index=range(1, 17), columns=range(1, 17))
print("Generated Random Numbers:")
print(random_numbers_df)

# Multiply the distance matrix by the random numbers
NEW_distance_matrix = distance_matrix * random_numbers

#Display NEW distance matrix
df_NEW_distance_matrix = pd.DataFrame(NEW_distance_matrix, index=range(1, 17), columns=range(1, 17))
print(df_NEW_distance_matrix)

# Convert the matrix into a symmetric matrix with diagonal elements zero
symmetric_matrix = np.triu(NEW_distance_matrix) + np.tril(NEW_distance_matrix.T, -1)

# Create a DataFrame
df_symmetric_matrix = pd.DataFrame(symmetric_matrix, index=range(1, 17), columns=range(1, 17))

# Print the DataFrame
print(df_symmetric_matrix)


# Create a model
model = gp.Model("TSP")

# Create decision variables
x = model.addVars(range(1, 17), range(1, 17), vtype=GRB.BINARY, name="x")

# Ensure x[i, j] is 0 when i = j
for i in range(1, 17):
    for j in range(1, 17):
        if i == j:
            model.addConstr(x[i, j] == 0)

# Ensuring that the graph is an undirected graph [ xij=xji ]
for i in range(1, 17):
    for j in range(1, 17):
        model.addConstr(x[i, j] == x[j, i])


# In[2]:


#Constraints


# In[3]:


# Ensure each node has exactly two incident edges
for i in range(1, 17):
    model.addConstr(gp.quicksum(x[i, j] for j in range(1, 17)) == 2, f"degree_{i}")


# In[4]:


# Eliminate cycles of length 3 (triangle subtours)
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            if i != j and j != k and i != k:
                model.addConstr(x[i, j] + x[j, k] + x[k, i] <= 2, f"no_triangle_subtour_{i}_{j}_{k}")


# In[5]:


# Eliminate cycles of length 4
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            for l in range(1, 17):
                if i != j and j != k and k != l and i != k and i != l and j != l:
                    model.addConstr(x[i, j] + x[j, k] + x[k, l] + x[l, i] <= 3, f"no_cycle_4_{i}_{j}_{k}_{l}")


# In[6]:


# Eliminate cycles of length 5
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(1, 17):
            for l in range(1, 17):
                for m in range(1, 17):
                    if i != j and j != k and k != l and l != m and i != k and i != l and i != m and j != l and j != m and k != m:
                        model.addConstr(x[i, j] + x[j, k] + x[k, l] + x[l, m] + x[m, i] <= 4, f"no_cycle_5_{i}_{j}_{k}_{l}_{m}")


# In[9]:


# Set objective: minimize the sum of distances * x[i, j]
model.setObjective(
    gp.quicksum(symmetric_matrix[i-1, j-1] * x[i, j] for i in range(1, 16) for j in range(1, 16)),
    GRB.MINIMIZE
)

# Optimize the model
model.optimize()


# In[10]:


# Print the solution
if model.status == GRB.OPTIMAL:
    for i in range(1, 17):
        for j in range(1, 17):
            if x[i, j].x > 0.5: #0.5 is arbitrary,the aim is to pick x[i,j]=1
                print(f"Edge ({i}, {j}) is selected in the optimal solution.")
else:
    print("No optimal solution found.")



# We have reached a TSP solution by eliminating all the subtours of length 3, 4 and 5.
# 
# 2-matching relaxation and adding cuts took much less time than 1-tree relaxation and adding cuts. 
# 
# For 2-matching, eliminating cycles with length less than or equal to 5 sufficed to get a solution for TSP.
# For 1-tree, elimination of cycles with length less than or equal to 4 did not suffice and I needed to add manual cuts. Since it took so many of them I just added the 2-matching constraint to terminate the search.

# In[12]:


# Given tour as a list of edges
tour = [
    (1, 2), (1, 8), (2, 1), (2, 3), (3, 2), (3, 16), (4, 8), (4, 15),
    (5, 6), (5, 15), (6, 5), (6, 7), (7, 6), (7, 9), (8, 1), (8, 4),
    (9, 7), (9, 10), (10, 9), (10, 12), (11, 14), (11, 16), (12, 10),
    (12, 13), (13, 12), (13, 14), (14, 11), (14, 13), (15, 4), (15, 5),
    (16, 3), (16, 11)
]

# Calculate total distance for the tour
total_distance = sum(symmetric_matrix[i-1, j-1] for i, j in tour) / 2

print("Total Distance for the Given Tour:", total_distance)


# In[ ]:




