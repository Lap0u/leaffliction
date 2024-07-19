import argparse
import Distribution
import Augmentation
import random
import cv2
from PIL import Image

AUGMENT_COUNT = 7


def flatten_dir(count_dir):
    """Flatten the count directory to a single dictionary"""
    classes = {}
    for key in count_dir:
        for subkey in count_dir[key]:
            classes[subkey] = count_dir[key][subkey]
    return classes


def augment_current(augmentation_id, target_image):
    """Augment the target image"""
    img = Image.open(target_image)
    cv2_img = cv2.imread(target_image)
    match (augmentation_id):
        case 0:  # Flip
            Augmentation.flip_and_save(img=img, path=target_image)
        case 1:  # Rotate
            Augmentation.rotate_and_save(img=img, path=target_image)
        case 2:  # Blur
            Augmentation.blur_and_save(img=img, path=target_image)
        case 3:  # Contrast
            Augmentation.add_contrast_and_save(img=cv2_img, path=target_image)
        case 4:  # Scale
            Augmentation.scale_and_save(img=img, path=target_image)
        case 5:  # Brightness
            Augmentation.add_brightness_and_save(img=cv2_img, path=target_image)


def balance_dataset(path, key, min_images, current_images):
    """Balance a single dataset, augmenting images as necessary"""
    print(path, key, min_images, current_images)
    required_augmentations = min_images - current_images
    available_augmentations = current_images * (AUGMENT_COUNT - 1)
    augmentation_array = random.sample(
        range(available_augmentations), required_augmentations
    )
    augmentation_array.sort()
    images = Distribution.fetch_images(path)
    images = [image for image in images if key in image]
    for augmentation_id in augmentation_array:
        target_image = images[augmentation_id // (AUGMENT_COUNT - 1)]
        augment_current(augmentation_id % (AUGMENT_COUNT - 1), target_image)


def balance_all_dataset(path, count_dir):
    """Balance the dataset by Augmenting images so that all classes have the same number of images"""
    classes = flatten_dir(count_dir)
    min_images = min([classes[key] for key in classes]) * AUGMENT_COUNT
    for key in classes:
        balance_dataset(path, key, min_images, classes[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument("-d", "--dataset", help="Dataset to balance", required=True)
    args = parser.parse_args()
    images = Distribution.fetch_images(args.dataset)
    count_dir = Distribution.count_images_per_directory(images)
    count_dir = Distribution.group_data_by_class(count_dir)
    balance_all_dataset(path=args.dataset, count_dir=count_dir)
