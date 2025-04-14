# model.py
import tensorflow as tf

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # Test the model with dummy dimensions (for example: 10 features, 2 classes)
    model = build_model((10,), 2)
    model.summary()
