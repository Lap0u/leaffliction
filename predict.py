import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import argparse

def predict(path_model, path_data):
	try:
		model = tf.keras.models.load_model(path_model)
	except ValueError:
		print(f"Error, no model available at {path_model}")
	
	train_images, validation_images = tf.keras.utils.image_dataset_from_directory(
	path_data,
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


if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Permet de charger le modele et de predire les donnees")

	parser.add_argument('path_model', default='model', help="Path vers le model a charger.")  # Argument positionnel
	parser.add_argument('--path_data', default='leaves/images/', help="Path vers les donnees a predire.")
	
	args = parser.parse_args()

	predict(args.path_model, args.path_data)