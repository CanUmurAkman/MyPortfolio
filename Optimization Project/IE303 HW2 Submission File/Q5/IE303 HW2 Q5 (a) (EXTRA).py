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


# In[8]:


#Display symmetric matrix
df_symmetric_matrix = pd.DataFrame(symmetric_matrix, index=range(1, 17), columns=range(1, 17))
print(df_symmetric_matrix)


# In[9]:


#Turn the Ulysses distance matrix into an asymmetric one by randomly changing distances.
# Generate random numbers for the lower triangle
random_lower_triangle = np.random.uniform(1, 2, size=(16, 16))

# Apply the random numbers to the lower triangle
asymmetric_matrix = symmetric_matrix.copy()
asymmetric_matrix[np.tril_indices(16, -1)] *= random_lower_triangle[np.tril_indices(16, -1)]

# Display the asymmetric matrix
df_asymmetric_matrix = pd.DataFrame(asymmetric_matrix, index=range(1, 17), columns=range(1, 17))
print("Asymmetric Distance Matrix:")
print(df_asymmetric_matrix)


# In[10]:


print(asymmetric_matrix[1,15]) 


# In[11]:


#Using (a) assignment constraints
# Number of nodes
num_nodes = asymmetric_matrix.shape[0]


# In[12]:


# Create a model
model = gp.Model("Asymmetric_TSP_Assignment")


# In[13]:


# Create decision variables
x = model.addVars(range(num_nodes), range(num_nodes), vtype=GRB.BINARY, name="x")


# In[14]:


# Ensure x[i, j] is 0 when i = j 
for i in range(num_nodes):
    model.addConstr(x[i, i] == 0)


# In[15]:


# Constraints: each node is visited once
for i in range(num_nodes):
    model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if i != j) == 1, f"visiting_{i}")


# In[16]:


# Constraints: each node is left once
for j in range(num_nodes):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if i != j) == 1, f"leaving_{j}")


# In[17]:


# Constraints: xij and xji cannot be 1 at the same time
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            model.addConstr(x[i, j] + x[j, i] <= 1, f'Directionality_{i}_{j}')


# In[18]:


# Set objective: minimize the sum of distances * x[i, j]
model.setObjective(
    gp.quicksum(asymmetric_matrix[i, j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)),
    GRB.MINIMIZE
)


# In[19]:


# Optimize the model
model.optimize()


# In[20]:


# Print the solution
if model.status == GRB.OPTIMAL:
    tour = [i for i in range(num_nodes) if any(x[i, j].x > 0.5 for j in range(num_nodes) if i != j)]
    
    print("Optimal Tour:")
    print(tour)
    
    selected_edges = [(i, j) for i in tour for j in range(num_nodes) if i != j and x[i, j].x > 0.5]
    print("Edges in the Optimal Tour:")
    for edge in selected_edges:
        print(f"Edge {edge[0]} - {edge[1]} (Distance: {asymmetric_matrix[edge[0], edge[1]]})")
    
    total_distance = sum(asymmetric_matrix[i, j] for i, j in zip(tour, tour[1:]))
    print(f"Total Distance: {total_distance}")
else:
    print("No optimal solution found.")


# CPU model: Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz
# Thread count: 2 physical cores, 4 logical processors, using up to 4 threads
# 
# Total Distance: 13129.591092666533
