# preprocessing.py (CORRIGÉ)
from tensorflow import keras

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def create_generators(data_dir):
    # Utilisez le chemin complet depuis keras
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,  
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,  # Même taille
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,  
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen