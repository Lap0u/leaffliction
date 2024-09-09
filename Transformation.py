import argparse
import os
from PIL import Image
import Distribution
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import numpy as np


def analyze_object():
    args = WorkflowInputs(
        images=["./leaves/images/Apple_Black_rot/image (1).JPG"],
        names="image",
        result="example_results_oneimage_file.csv",
        outdir=".",
        writeimg=False,
        # debug="plot",
    )
    # Set debug to the global parameter
    pcv.params.debug = args.debug
    # Change display settings
    pcv.params.dpi = 100
    pcv.params.text_size = 20
    pcv.params.text_thickness = 20
    img, path, filename = pcv.readimage(filename=args.image)
    # Fill in small objects if the above threshold looks like there are "holes" in the leaves
    thresh1 = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )
    print(img.shape)
    roi1 = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    a_fill_image = pcv.fill(bin_img=thresh1, size=50)
    a_fill_image = pcv.fill_holes(a_fill_image)
    kept_mask = pcv.roi.filter(mask=a_fill_image, roi=roi1, roi_type="partial")
    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    pcv.plot_image(analysis_image)


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
        "-src", "--source", help="Directory or file to transform", required=True
    )
    parser.add_argument("-dst", "--destination", help="Path to save the file to")
    args = parser.parse_args()
    if os.path.isdir(args.source) and args.destination is None:
        print("Warning: Destination path is required for folders")
        exit(1)
    mask_img(args.source)
    analyze_object()
    if os.path.isfile(args.source):
        transform_img(args.source)
    else:
        transform_folder(args.source, args.destination)
