# model.py - Correction finale
from tensorflow import keras

IMG_HEIGHT = 128 
IMG_WIDTH = 128  
NUM_CLASSES = 4

def build_model():
    # CORRECTION : EfficientNetB0 vient de keras.applications
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)  
    )
    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model