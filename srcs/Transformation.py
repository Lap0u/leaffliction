import argparse
import os
from PIL import Image
import Distribution
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import numpy as np


def analyze_with_roi(image):
    """"""
    # Définir une ROI (ici un rectangle autour de la plante)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(
        img=image, x=10, y=10, h=10, w=10
    )


def transformation_task(path: str):
    """"""
    # retriving original image
    image, path, image_name = pcv.readimage(filename=path)
    # original to gray for next operation
    gray_img = pcv.rgb2gray(rgb_img=image)
    # gray to background black and content white and inverse
    bin_img_light = pcv.threshold.binary(
        gray_img, threshold=125, object_type="light"
    )
    bin_img_dark = pcv.threshold.binary(
        gray_img, threshold=125, object_type="dark"
    )
    # pcv.plot_image(bin_img_light)
    # pcv.plot_image(bin_img_dark)
    # apply fill_holes to remove noise in binary image
    # bin_img_light = pcv.fill_holes(bin_img_light)
    bin_img_dark = pcv.fill_holes(bin_img_dark)
    # pcv.plot_image(bin_img_light)
    # pcv.plot_image(bin_img_dark)
    # Apply gaussian blur to reduce noise
    gaussian_img_light = pcv.gaussian_blur(
        img=bin_img_light, ksize=(15, 15), sigma_x=0, sigma_y=None
    )
    gaussian_img_dark = pcv.gaussian_blur(
        img=bin_img_dark, ksize=(15, 15), sigma_x=0, sigma_y=None
    )
    # pcv.plot_image(gaussian_img_light)
    # pcv.plot_image(gaussian_img_dark)

    # apply white and black mask
    masked_image_light = pcv.apply_mask(
        img=image, mask=bin_img_light, mask_color="white"
    )
    masked_image_dark = pcv.apply_mask(
        img=image, mask=bin_img_dark, mask_color="black"
    )
    # pcv.plot_image(masked_image_light)
    # pcv.plot_image(masked_image_dark)

    # compute RoI (Region of Interest) object
    mask = pcv.naive_bayes_classifier(
        rgb_img=image, pdf_file="./machine_learning.txt"
    )
    # analyze_with_roi(image=image)


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


def mask_img(path, destination=None):
    """Apply mask to the image"""
    image = cv2.imread(path)
    cv2.imshow("Original", image)
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_green, high_green)
    cv2.imshow("Mask", mask)
    new_path = "./test.jpg"
    cv2.imwrite(new_path, mask)


def transform_img(path, destination=None):
    """Apply all transformation to the image and save it as necessary"""
    image = Image.open(path)
    threshold_filter(path, image, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument(
        "-src",
        "--source",
        help="Directory or file to transform",
        required=True,
    )
    parser.add_argument(
        "-dst", "--destination", help="Path to save the file to"
    )
    args = parser.parse_args()
    if os.path.isdir(args.source) and args.destination is None:
        print("Warning: Destination path is required for folders")
        exit(1)
    if os.path.isfile(args.source):
        transformation_task(args.source)
    else:
        transform_folder(args.source, args.destination)
