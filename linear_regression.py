
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
x = np.random.rand(100, 1) * 100  # 100 random numbers scaled to 0-100
y = 3 * x + np.random.randn(100, 1) * 30 + 50  # y = 3x + noise + 50

# Create a linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values
x_new = np.linspace(0, 100, 100).reshape(100, 1)
y_pred = model.predict(x_new)

# Plotting the results
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_new, y_pred, color='red', label='Regression line')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

