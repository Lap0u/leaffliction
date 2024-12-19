import argparse
import os
from PIL import Image
import Distribution
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm.auto import tqdm
pcv.params.sample_label="leaf"


def add_image_to_plot(axes: plt.axes, pos_x:int, pos_y:int, title:str, img: np.ndarray):
    axes[pos_x, pos_y].imshow(img)
    axes[pos_x, pos_y].set_title(title)
    axes[pos_x, pos_y].axis('off')


def add_pseudoland_to_image(posMarks, img):
    for pos, color in zip(posMarks, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        pos = pos.flatten()
        X = [int(pos[i]) for i in range(0, len(pos), 2)]
        Y = [int(pos[i]) for i in range(1, len(pos), 2)]
        for x, y in zip(X, Y):
            cv2.circle(img, (x, y), radius=5, color=color, thickness=-1)
    return img


def add_point_to_img(posBranch, img):
    for pos_y, y in enumerate(posBranch):
        for pos_x, x in enumerate(y):
            if x != 0:
                cv2.circle(img, (pos_x, pos_y), radius=5, color=(255,0,0), thickness=-1)
    return img


def transformation_task(path: str, plot: bool):
    """
    Apply all transformation to the image and save it as necessary
    """
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    img, path, image_name = pcv.readimage(filename=path)

    # pcv.params.debug = "plot"
    colorspaces = pcv.visualize.colorspaces(rgb_img=img, original_img=True)

    a = pcv.rgb2gray_lab(rgb_img=img, channel="a")


    a_thresh = pcv.threshold.binary(gray_img=a, threshold=120, object_type='dark')

    roi1 = pcv.roi.rectangle(img=a_thresh, x=35, y=10, h=245, w=200)

    kept_mask = pcv.roi.filter(mask=a_thresh, roi=roi1, roi_type='partial')

    mask_dilated = pcv.dilate(gray_img=kept_mask, ksize=3, i=1)

    mask_fill = pcv.fill(bin_img=mask_dilated, size=30)
    mask_fill = pcv.fill_holes(bin_img=mask_fill)

    skeleton = pcv.morphology.skeletonize(mask=mask_fill)

    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=mask_fill)

    img_tips_pts = add_point_to_img(pcv.morphology.find_tips(skel_img=pruned_skel, mask=mask_fill, label="default"), pruned_skel.copy())

    img_branch_pts = add_point_to_img(pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=mask_fill, label="default"), pruned_skel.copy())

    analyse_img = pcv.analyze.size(img=img, labeled_mask=mask_fill, label="default")

    img_pseudolandmarks_x = add_pseudoland_to_image(pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask_fill), img.copy())
    img_pseudolandmarks_y = add_pseudoland_to_image(pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask_fill), img.copy())

    if plot:
        add_image_to_plot(axes, 0, 0, "Original", img)
        add_image_to_plot(axes, 0, 1, "'A' filter applied", a)
        add_image_to_plot(axes, 0, 2, "'A tresh' filter applied", a_thresh)
        add_image_to_plot(axes, 0, 3, "Dilation Filtering", mask_dilated)
        add_image_to_plot(axes, 1, 0, "Fill Holes applied", mask_fill)
        add_image_to_plot(axes, 1, 1, "Skeleteton transorm", skeleton)
        add_image_to_plot(axes, 1, 2, "Pruned skeleton transform", pruned_skel)
        add_image_to_plot(axes, 1, 3, "Find tips in skeleton", img_tips_pts)
        add_image_to_plot(axes, 2, 0, "Find branch in skeleton", img_branch_pts)
        add_image_to_plot(axes, 2, 1, "Analyse Image", analyse_img)
        add_image_to_plot(axes, 2, 2, "Pseudolandmarks X", img_pseudolandmarks_x)
        add_image_to_plot(axes, 2, 3, "pseudolandmarks Y", img_pseudolandmarks_y)
        plt.show()
    plt.close()
    return pruned_skel, img_tips_pts, img_branch_pts, analyse_img, img_pseudolandmarks_x, img_pseudolandmarks_y



def transform_folder(path, destination):
    """Transform all images in the folder"""
    images = Distribution.fetch_images(path)
    for image in tqdm(images):
        transform_img(image, destination)


def register_transformed_imgs(origin_path: str, to_save_image_list: List[np.ndarray], destination_path: str):
    origin_dir = "/" + origin_path[:origin_path.rfind('/')]
    if not os.path.exists(destination_path + origin_dir):
        os.makedirs(destination_path + origin_dir)
    suffix = ["_skeleton", "_tips_point", "_branch_point", "_analysed_shape", "_pseudolandmarksX", "_pseudolandmarksY"]
    for image, suf in zip(to_save_image_list, suffix):
        name = destination_path + origin_path[:-4] + suf + ".jpg"
        name = ''.join(name.split())
        # print(name)
        cv2.imwrite(name, image)


def transform_img(path, destination=None):
    """Apply all transformation to the image and save it as necessary"""
    modified_img_list = transformation_task(path, False)
    register_transformed_imgs(path, modified_img_list, destination)


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
        transformation_task(args.source, True)
    else:
        transform_folder(args.source, args.destination)
