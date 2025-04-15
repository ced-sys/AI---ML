import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**2

def derivative(x):
    return 2*x

def gradient_descent(starting_point, learning_rate, num_iterations):
    x=starting_point
    history=[x]

    for _ in range(num_iterations):
        grad=derivative(x)
        x=x-learning_rate* grad
        history.append(x)

    return x, history

starting_point=4.0
learning_rate=0.1
num_iterations=50

optimal_x, history=gradient_descent(starting_point, learning_rate, num_iterations)

print(f"Starting point: {starting_point}")
print(f"Optimal x:{optimal_x}")
print(f"Minimum value: {function(optimal_x)}")

x_values=np.linspace(-5, 5, 100)
y_values=function(x_values)

plt.plot(x_values, y_values, 'b-', label='f(x) =x^2')
plt.plot(history, [function(x) for x in history], 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()