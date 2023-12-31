from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from part1 import load_data
import os
import pickle

X_train, X_test, y_train, y_test = load_data(test_size=0.30)

print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])

model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-8,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}

model = MLPClassifier(**model_params)
print("[*] Training the model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

print(classification_report(y_test, y_pred))

if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("mlp_classifier.model", "wb"))

