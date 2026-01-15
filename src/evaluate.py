import numpy as np
# Importation directe depuis Keras (TensorFlow >=2.13)
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import create_generators

DATA_DIR = "data/"
MODEL_PATH = "models/simple_cnn_model.h5"

def evaluate():
    _, _, test_gen = create_generators(DATA_DIR)
    model = load_model(MODEL_PATH)

    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    print("📊 Classification Report :")
    print(classification_report(
        y_true,
        y_pred,
        target_names=test_gen.class_indices.keys()
    ))

    print("📉 Confusion Matrix :")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
