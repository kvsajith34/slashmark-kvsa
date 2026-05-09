"""
model.py — CNN architecture for the Cats vs. Dogs classifier.

I kept things deliberately readable here. The network is intentionally
small enough to train on a laptop GPU (or even CPU for a few epochs),
while still being deep enough to learn useful visual features.

Architecture at a glance
------------------------
Input (150 × 150 × 3)
  → Conv Block 1  : Conv2D(32)  + BatchNorm + MaxPool
  → Conv Block 2  : Conv2D(64)  + BatchNorm + MaxPool
  → Conv Block 3  : Conv2D(128) + BatchNorm + MaxPool
  → Conv Block 4  : Conv2D(256) + BatchNorm + MaxPool
  → GlobalAvgPool  (replaces a Flatten + big Dense — far fewer params)
  → Dense(256) + Dropout(0.5)
  → Dense(1, sigmoid)  — binary output: 0 = cat, 1 = dog
"""

from tensorflow.keras import layers, models, regularizers


def build_model(input_shape=(150, 150, 3), l2_lambda=1e-4):
    """
    Build and return the compiled CNN model.

    Parameters
    ----------
    input_shape : tuple
        (height, width, channels) expected by the network.
    l2_lambda : float
        L2 weight-decay applied to Conv layers — helps prevent overfitting
        on smaller datasets.

    Returns
    -------
    model : keras.Model
        Compiled model, ready for .fit().
    """

    def conv_block(x, filters, pool=True):
        """One repeatable building block so the main graph stays clean."""
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
        )(x)
        x = layers.BatchNormalization()(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=2)(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # --- Feature extraction ---
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    # GlobalAveragePooling: one mean value per feature map → (batch, 256)
    # Much cheaper than Flatten → Dense and less prone to overfitting.
    x = layers.GlobalAveragePooling2D()(x)

    # --- Classifier head ---
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(0.5)(x)   # drop half the units during training
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="CatDog_CNN")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
