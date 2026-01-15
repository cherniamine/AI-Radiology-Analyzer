# train.py (updated)
from preprocessing import create_generators
from model import build_model
# Updated import style for Keras 3
from keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = "data/"
MODEL_PATH = "models/simple_cnn_model.h5"

def train():
    train_gen, val_gen, _ = create_generators(DATA_DIR)
    model = build_model()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    print("✅ Training terminé. Modèle sauvegardé.")

if __name__ == "__main__":
    train()