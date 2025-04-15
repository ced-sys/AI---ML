import numpy as np

# Define X and y
X = np.array([[1, x] for x in range(1, 11)])  # Adding a column of 1s for intercept
y = np.array([2, 4, 5, 4, 5, 6, 7, 8, 9, 10])

# Compute OLS manually
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Print results
print("Intercept (c):", theta[0])
print("Slope (m):", theta[1])
