import argparse
import os
from PIL import Image
import Distribution
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import numpy as np
import matplotlib.pyplot as plt
pcv.params.sample_label="leaf"


def analyze_with_roi(image):
    """"""
    # Définir une ROI (ici un rectangle autour de la plante)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(
        img=image, x=10, y=10, h=10, w=10
    )


def transformation_task(path: str):
    """"""
    # retriving original image
    img, path, image_name = pcv.readimage(filename=path)

    #Select the better channel that maximizes the difference 
    # between the plant and the background pixels
    # pcv.params.debug = "plot"
    colorspaces = pcv.visualize.colorspaces(rgb_img=img, original_img=False)

    #we choose a or b
    a = pcv.rgb2gray_lab(rgb_img=img, channel="a")

    a_thresh = pcv.threshold.binary(gray_img=a, threshold=120, object_type='dark')

    roi1 = pcv.roi.rectangle(img=a_thresh, x=35, y=10, h=245, w=200)

    kept_mask = pcv.roi.filter(mask=a_thresh, roi=roi1, roi_type='partial')

    mask_dilated = pcv.dilate(gray_img=kept_mask, ksize=3, i=1)

    mask_fill = pcv.fill(bin_img=mask_dilated, size=30)
    mask_fill = pcv.fill_holes(bin_img=mask_fill)

    skeleton = pcv.morphology.skeletonize(mask=mask_fill)

    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=mask_fill)

    tip_pts_mask = pcv.morphology.find_tips(skel_img=pruned_skel, mask=None, label="default")

    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=mask_fill, label="default")

    shape_img = pcv.analyze.size(img=img, labeled_mask=mask_fill, label="default")

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask_fill)
    top, bottom, center_v = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask_fill)


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
