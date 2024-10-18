import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="This program train model to predict leaf class")
    parser.add_argument("path_data", default="leaves/images/", help="The path to data to train with")
    args = parser.parse_args()

    train_images, validation_images = keras.utils.image_dataset_from_directory(
        args.path_data,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(32, 32),
        seed=42,
        validation_split=0.2,
        subset="both",
        interpolation="bilinear",
        follow_links=False,
    )

print(train_images)
# class_names = [
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# ]

# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
# model.add(keras.layers.MaxPooling2D((2, 2)))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(64, activation="relu"))
# model.add(keras.layers.Dense(10))
# model.summary()

# loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optim = keras.optimizers.Adam(learning_rate=0.001)
# metrics = ["accuracy"]
# model.compile(optimizer=optim, loss=loss, metrics=metrics)

# epochs = 100

# model.fit(train_images, epochs=epochs, verbose=2)
# print("\033[96mModel train complete\033[0m")
# print("\033[92mNow evaluating model...")
# model.evaluate(validation_images, verbose=2)
# print("\033[0m")
# if not os.path.exists('model'):
# 	os.mkdir("model")
# model.save("model/model"+datetime.now().strftime("_%m-%d_%H:%M")+".keras")

# Ici enregister dans un .zip les images ayant servis a l'entrainement!
# Enregistrer separement les images du training et celles de la prediction!
# Et le model precedement entrainer et enregistrer!
