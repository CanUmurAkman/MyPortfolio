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


# In[5]:


# Greedy algorithm to find Hamiltonian tour
current_city = 1  # Starting from city 1
T = []  # Tour
selected_cities = set([current_city])

while len(T) < 16:
    max_weight = -1
    next_city = -1

    for j in range(1, 17):
        if j != current_city and j not in selected_cities:
            if symmetric_matrix[current_city-1, j-1] > max_weight:
                max_weight = symmetric_matrix[current_city-1, j-1]
                next_city = j

    if next_city != -1:
        T.append((current_city, next_city))
        selected_cities.add(next_city)
        current_city = next_city
    else:
        break


# In[9]:


# Output selected edges
print("Selected Edges:")
for edge in T:
    print(f"Edge {edge[0]} - {edge[1]}")

# Output Hamiltonian tour
print("Tour:")
for edge in T:
    print(f"Edge {edge[0]} - {edge[1]}")
print(f"Edge {T[-1][1]} - {T[0][0]}")  # Closing the loop

# Calculate total distance
total_distance = sum(symmetric_matrix[edge[0]-1, edge[1]-1] for edge in T)/2
print(f"Total Distance: {total_distance}")


# 13749 (Greedy Algorithm Total Distance) - 11544 (Optimal Distance) = 2205
# Well, greedy algorithm is far away from the optimal distance, but not too far away.
