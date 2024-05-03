import matplotlib.pyplot as plt  # Make sure this line is at the top of your script
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import pandas as pd


# Sample DataFrame creation
data = {
    'Age': [25, 45, 35, 32, 41],
    'Job': ['admin', 'technician', 'entrepreneur', 'admin', 'technician'],
    'Marital': ['single', 'married', 'married', 'single', 'divorced'],
    # 'Education': ['tertiary', 'secondary', 'tertiary', 'primary', 'secondary'],
    'Balance': [500, 600, -200, 300, 400],
    'Outcome': ['no', 'yes', 'no', 'no', 'yes']
}
df = pd.DataFrame(data)

# Encoding categorical data
df_encoded = pd.get_dummies(df.drop('Outcome', axis=1))
df_encoded['Outcome'] = df['Outcome'].apply(lambda x: 1 if x == 'yes' else 0)

# Splitting dataset
X = df_encoded.drop('Outcome', axis=1)
y = df_encoded['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree classifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=X_train.columns, class_names=['No', 'Yes'])
plt.show()
