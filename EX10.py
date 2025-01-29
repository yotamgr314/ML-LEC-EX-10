from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import sklearn.metrics as metric
import numpy as np

# קביעת seed כדי להבטיח חזרתיות
np.random.seed(42)

# פונקציה להערכת המודל
def evaluate_model(model, X_test, y_true, model_name="Model"):
    y_pred = model.predict(X_test)
    accuracy = metric.accuracy_score(np.array(y_true).flatten(), np.array(y_pred).flatten())
    print(f"{model_name} Prediction: {y_pred}")
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    if hasattr(model, 'coefs_'):
        print("Learned Weights Shapes:", [coef.shape for coef in model.coefs_])
        print("Learned Weights:", model.coefs_)
    elif hasattr(model, 'coef_'):
        print("Learned Weights:", model.intercept_, model.coef_)
    print("-" * 50)

X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 1, 1, 0]

ptn = Perceptron(max_iter=500, random_state=42)
ptn.fit(X_training, y_training)
evaluate_model(ptn, X_training, y_training, "Perceptron OR Gate")

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1, 1), activation='logistic', max_iter=1000, random_state=42)
mlp.fit(X_training, y_training)
mlp.coefs_[0] = np.array([[4.1486074], [4.14636493]])
mlp.coefs_[1] = np.array([[7.43468985]])
mlp.coefs_[2] = np.array([[15.53567128]])
evaluate_model(mlp, X_training, y_training, "MLP OR Gate")

X_training = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
y_training = [1, 1, 1, -1]
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2,), activation='tanh', max_iter=1000, random_state=42)
mlp.fit(X_training, y_training)
evaluate_model(mlp, X_training, y_training, "MLP Bipolar OR Gate")

X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 0, 0, 0]
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2,), activation='logistic', max_iter=1000, random_state=42)
mlp.fit(X_training, y_training)
evaluate_model(mlp, X_training, y_training, "MLP AND Gate")

y_training = [1, -1, -1, -1]
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2,), activation='tanh', max_iter=1000, random_state=42)
mlp.fit(X_training, y_training)
evaluate_model(mlp, X_training, y_training, "MLP Bipolar AND Gate")

X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [[1, 1], [1, 1], [1, 1], [0, 1]]
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2, 2), activation='logistic', max_iter=1000, random_state=42)
mlp.fit(X_training, y_training)
evaluate_model(mlp, X_training, y_training, "MLP Two Output Neurons")

X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [1, 0, 0, 1]
mlp_xor = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3, 3), activation='tanh', max_iter=1000, random_state=42)
mlp_xor.fit(X_training, y_training)
evaluate_model(mlp_xor, X_training, y_training, "MLP XOR Gate - Improved")

X_training = [[1, 1, 0], [1, -1, -1], [-1, 1, 1], [-1, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 1]]
y_training = [[1, 0], [0, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]]
mlp_3in_2out = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 4, 4), activation='tanh', max_iter=1000, random_state=42)
mlp_3in_2out.fit(X_training, y_training)
evaluate_model(mlp_3in_2out, X_training, y_training, "MLP 3 Input - 2 Output - Improved")
