from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.001
iterations = 100

# plot predictions for every iteration?
do_plot = True # initially true

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
print('The final theta values found:')
print(theta_final, '\n')

new_samples = [[1650,3], [3000,4]] # new ndarray

# normalize and add bias:
# Make sure to apply the same preprocessing that was applied to the training data
new_samples = (new_samples - mean_vec) / std_vec
column_of_ones = np.ones((new_samples.shape[0], 1))
new_samples = np.append(column_of_ones, new_samples, axis=1)

print('New (Normalized) Samples:')
print(new_samples, '\n')

# Calculate the hypothesis for each sample, using the trained parameters theta_final
hyp = []
for i in range(len(new_samples)):
    hyp.append(calculate_hypothesis(new_samples, theta_final, i))

# Print the predicted prices for the two samples
print('Area of 1650, 3 bedroom house price: ', hyp[0])
print('Area of 3000, 4 bedroom house price: ', hyp[1])
########################################