import argparse
import os
from PIL import Image
import Distribution


def threshold_filter(path, image, destination=None):
    """Apply threshold filter to the image"""
    image = image.convert("L")
    image = image.point(lambda x: 0 if x < 100 else 255)
    new_image = path.replace(".JPG", "_thresholded.JPG")
    if destination:
        get_folder_path = path.split("/")
        get_folder_path.pop()
        new_path = destination + "/" + "/".join(get_folder_path)
        os.makedirs(new_path, exist_ok=True)
        print("Saving to:", destination + "/" + new_image)
        image.save(destination + "/" + new_image)
    else:
        image.show()


def transform_folder(path, destination):
    """Transform all images in the folder"""
    images = Distribution.fetch_images(path)
    for image in images:
        transform_img(image, destination)


def transform_img(path, destination=None):
    """Apply all transformation to the image and save it as necessary"""
    image = Image.open(path)
    threshold_filter(path, image, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument(
        "-src", "--source", help="Directory or file to transform", required=True
    )
    parser.add_argument("-dst", "--destination", help="Path to save the file to")
    args = parser.parse_args()
    if os.path.isdir(args.source) and args.destination is None:
        print("Warning: Destination path is required for folders")
        exit(1)
    if os.path.isfile(args.source):
        transform_img(args.source)
    else:
        transform_folder(args.source, args.destination)
