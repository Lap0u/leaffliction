import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import cv2


def blur_and_save(img, path):
    """Blur the image and save it"""
    blurred_img = img.filter(ImageFilter.BLUR)
    return save_transformation(
        img=blurred_img, path=path, transformation="Blurred"
    )


def rotate_and_save(img, path):
    """Rotate the image and save it"""
    degrees = random.randint(-45, 45)
    rotated_img = img.rotate(degrees)
    return save_transformation(
        img=rotated_img, path=path, transformation="Rotated"
    )


def save_transformation(img, path, transformation):
    """Save the transformed pillow image"""
    new_path = path.replace(".JPG", f"_{transformation}.JPG")
    img.save(new_path)
    return new_path


def save_transformation_cv2(img, path, transformation):
    """Save the transformed cv2 image"""
    new_path = path.replace(".JPG", f"_{transformation}.JPG")
    cv2.imwrite(new_path, img)
    return new_path


def display_augmented_images(paths):
    """Display the augmented images"""
    fig = plt.figure(figsize=(12, 8))
    columns = len(paths)
    for i in range(columns):
        fig.add_subplot(1, columns, i + 1)
        img = Image.open(paths[i])
        plt.axis("off")
        title = paths[i].split("/")[-1].split("_")[-1].split(".")[0]
        plt.title(title)
        plt.imshow(img)
    plt.show()


def add_contrast_and_save(img, path):
    """Add contrast to the image and save it"""
    contrast = 2
    img = cv2.convertScaleAbs(img, alpha=contrast)
    return save_transformation_cv2(
        img=img, path=path, transformation="Contrast"
    )


def scale_and_save(img, path):
    """Scale the image and save it"""
    width, height = img.size  # Get dimensions

    left = width * 0.15
    top = width * 0.15
    right = width * 0.85
    bottom = width * 0.85
    img = img.crop((left, top, right, bottom))
    return save_transformation(img=img, path=path, transformation="Scaled")


def add_brightness_and_save(img, path):
    """Add brightness to the image and save it"""
    brightness = 100
    img = cv2.convertScaleAbs(img, beta=brightness)
    return save_transformation_cv2(
        img=img, path=path, transformation="Brigthness"
    )


def flip_and_save(img, path):
    """Flip the image and save it"""
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return save_transformation(
        img=flipped_img, path=path, transformation="Flipped"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument("-f", "--file", help="File to augment", required=True)
    args = parser.parse_args()
    img = Image.open(args.file)
    cv2_img = cv2.imread(args.file)
    augmented_paths = []
    augmented_paths.append(args.file)
    augmented_paths.append(flip_and_save(img=img, path=args.file))
    augmented_paths.append(rotate_and_save(img=img, path=args.file))
    augmented_paths.append(blur_and_save(img=img, path=args.file))
    augmented_paths.append(add_contrast_and_save(img=cv2_img, path=args.file))
    augmented_paths.append(scale_and_save(img=img, path=args.file))
    augmented_paths.append(
        add_brightness_and_save(img=cv2_img, path=args.file)
    )
    display_augmented_images(paths=augmented_paths)
