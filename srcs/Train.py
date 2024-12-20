import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tensorflow.keras import layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program train model to predict leaf class"
    )
    parser.add_argument(
        "path_data",
        default="data/leaves/images/",
        help="The path to data to train with",
    )
    args = parser.parse_args()

    train_images, validation_images = keras.utils.image_dataset_from_directory(
        args.path_data,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(64, 64),
        seed=42,
        validation_split=0.2,
        subset="both",
        interpolation="bilinear",
        follow_links=False,
    )

callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=2,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)
model = keras.models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]
model.compile(optimizer=optim, loss=loss, metrics=metrics)

epochs = 100

model.fit(
    train_images,
    validation_data=validation_images,
    epochs=epochs,
    verbose=2,
    callbacks=[callback],
)
print("\033[96mModel train complete\033[0m")
print("\033[92mNow evaluating model...")
model.evaluate(validation_images, verbose=2)
print(validation_images)
print("\033[0m")
if not os.path.exists("model"):
    os.mkdir("model")
model.save("model/model" + datetime.now().strftime("_%m-%d_%H:%M") + ".keras")

# Ici enregister dans un .zip les images ayant servis a l'entrainement!
# Enregistrer separement les images du training et celles de la validation!
# Et le model precedement entrainer et enregistrer!
