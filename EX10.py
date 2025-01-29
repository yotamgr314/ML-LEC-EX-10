# ייבוא הספריות הנדרשות
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import sklearn.metrics as metric

# פונקציה לאימון ותצוגה של תוצאות עבור Perceptron
def train_perceptron(X_train, y_train, X_test, y_true):
    ptn = Perceptron(max_iter=500)
    ptn.fit(X_train, y_train)
    y_pred = ptn.predict(X_test)
    accuracy = metric.accuracy_score(y_true, y_pred)
    print("\nPerceptron Prediction:", y_pred)
    print("Perceptron Accuracy:", accuracy)
    print("Perceptron Weights:", ptn.intercept_, ptn.coef_)
    return accuracy

# פונקציה לאימון ותצוגה של תוצאות עבור MLP
def train_mlp(X_train, y_train, X_test, y_true, hidden_layer_sizes=(1,1), activation='logistic', solver='lbfgs'):
    mlp = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = metric.accuracy_score(y_true, y_pred)
    print("\nMLP Prediction:", y_pred)
    print("MLP Accuracy:", accuracy)
    print("MLP Weights Shapes:", [coef.shape for coef in mlp.coefs_])
    print("MLP Weights:", mlp.coefs_)
    return accuracy

# 1. OR Gate
print("\n### OR Gate ###")
X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 1, 1, 0]
X_testing = X_training
y_true = y_training

train_perceptron(X_training, y_training, X_testing, y_true)
train_mlp(X_training, y_training, X_testing, y_true)

# 2. Bipolar OR Gate
print("\n### Bipolar OR Gate ###")
X_training = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
y_training = [1, 1, 1, -1]
X_testing = X_training
y_true = y_training

train_perceptron(X_training, y_training, X_testing, y_true)
train_mlp(X_training, y_training, X_testing, y_true)

# 3. AND Gate
print("\n### AND Gate ###")
X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 0, 0, 0]
X_testing = X_training
y_true = y_training

train_perceptron(X_training, y_training, X_testing, y_true)
train_mlp(X_training, y_training, X_testing, y_true)

# 4. Bipolar AND Gate
print("\n### Bipolar AND Gate ###")
X_training = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
y_training = [1, -1, -1, -1]
X_testing = X_training
y_true = y_training

train_perceptron(X_training, y_training, X_testing, y_true)
train_mlp(X_training, y_training, X_testing, y_true)

# 5. Two Output Neurons (צפוי להוביל לבעיה)
print("\n### Two Output Neurons ###")
X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [[1, 1], [1, 1], [1, 1], [0, 1]]
X_testing = X_training
y_true = y_training

try:
    train_mlp(X_training, y_training, X_testing, y_true)
except Exception as e:
    print("⚠ שגיאה במודל עם שני נוירונים ביציאה:", e)

# 6. XOR Gate (בדיקת מינימום שכבות חבויות נדרש)
print("\n### XOR Gate ###")
X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 0, 0, 1]
X_testing = X_training
y_true = y_training

train_mlp(X_training, y_training, X_testing, y_true, hidden_layer_sizes=(2,))

# 7. Neural Network with 3 input and 2 output neurons
print("\n### Neural Network with 3 Inputs and 2 Outputs ###")
X_training = [[1, 1, 0], [1, -1, -1], [-1, 1, 1], [-1, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 1]]
y_training = [[1, 0], [0, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]]
X_testing = X_training
y_true = y_training

train_mlp(X_training, y_training, X_testing, y_true, hidden_layer_sizes=(3, 2), activation='tanh', solver='adam')
