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

#Display symmetric matrix
df_symmetric_matrix = pd.DataFrame(symmetric_matrix, index=range(1, 17), columns=range(1, 17))
print(df_symmetric_matrix)

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

print(asymmetric_matrix[1,15]) 


# In[2]:


#Using (a) assignment constraints
# Number of nodes
num_nodes = asymmetric_matrix.shape[0]

# Create a model
model = gp.Model("Asymmetric_TSP_Assignment")

# Create decision variables
x = model.addVars(range(num_nodes), range(num_nodes), vtype=GRB.CONTINUOUS, name="x")

# Ensure x[i, j] is 0 when i = j 
for i in range(num_nodes):
    model.addConstr(x[i, i] == 0)

# Constraints: each node is left once
for i in range(num_nodes):
    model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if i != j) == 1, f"visiting_{i}")

# Constraints: each node is visited once
for j in range(num_nodes):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if i != j) == 1, f"leaving_{j}")

# Constraints: xij and xji cannot be 1 at the same time
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            model.addConstr(x[i, j] + x[j, i] <= 1, f'Directionality_{i}_{j}')


# In[3]:


# Between 0 and 1
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            model.addConstr(x[i, j]<= 1, f"one")
            model.addConstr(x[i, j]>=0, f"zero")


# In[4]:


#Using (b) assignment and MTZ contsraints
#Add the MTZ constraints
# Define new integer variables ui for each node
u = model.addVars(range(num_nodes), vtype=GRB.INTEGER, name="u")


# In[5]:


# Set lower bounds and upper bounds for u variables
for i in range(num_nodes):
    u[i].lb = 1  # Lower bound to ensure positive integers
    u[i].ub = num_nodes # Upper bound to limit the values (the last position is 16)


# In[6]:


# Set u[0] = 1
u[0].lb = 1
u[0].ub = 1


# In[7]:


# MTZ constraints
for i in range(num_nodes):
    for j in range(2, num_nodes):
        if i != j:
            model.addConstr(u[i] - u[j] + num_nodes * x[i, j] <= num_nodes - 1, f'MTZ_{i}_{j}')


# In[8]:


# Additional constraint to ensure the sum of u[i] is equal to 136
model.addConstr(gp.quicksum(u[i] for i in range(num_nodes)) == 136, "Sum_of_u")


# In[9]:


# Set objective: minimize the sum of distances * x[i, j]
model.setObjective(
    gp.quicksum(asymmetric_matrix[i, j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)),
    GRB.MINIMIZE
)


# In[10]:


# Optimize the model after adding the MTZ constraints
model.optimize()


# In[11]:


# Print the solution
if model.status == GRB.OPTIMAL:
    # Print the optimal values of the decision variables and u variables
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and x[i, j].x > 0.5:
                print(f"x[{i}, {j}] = {x[i, j].x}")
    
    for i in range(num_nodes):
        print(f"u[{i}] = {u[i].x}")

    # Calculate the total distance
    total_distance = sum(asymmetric_matrix[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j and x[i, j].x > 0.5)
    print(f"Total Distance: {total_distance}")
else:
    print("No optimal solution found.")


# CPU model: Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz
# Thread count: 2 physical cores, 4 logical processors, using up to 4 threads
# 
# Total Distance: 9560.758794129226
# 
# Cutting planes:
#   Gomory: 3
#   
#  3 CPs!
