import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
# Feature set
X = np.r_[np.random.randn(100, 2) - [2, 2], np.random.randn(100, 2) + [2, 2]]
# Labels
Y = np.array([0] * 100 + [1] * 100)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Create a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict test set
Y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter', edgecolors='k', alpha=0.7)
ax = plt.gca()
x_vals = np.array(ax.get_xlim())
y_vals = -(x_vals * model.coef_[0][0] + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_vals, y_vals, '--', c="red")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
