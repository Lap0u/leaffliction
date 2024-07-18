import argparse
import matplotlib.pyplot as plt
from PIL import Image


def display_img(img):
    """Display the image"""
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_transformation(img, path, transformation):
    """Save the transformed image"""
    pass


def flip_and_save(img):
    """Flip the image and save it"""
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    save_transformation(img=flipped_img, path="flipped.jpg", transformation="Flipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument("-f", "--file", help="File to augment", required=True)
    args = parser.parse_args()
    img = Image.open(args.file)
    flip_and_save(img=img, path=args.file)
    rotate_and_save(img=img, path=args.file)
    blur_and_save(img=img, path=args.file)
    add_contrast_and_save(img=img, path=args.file)
    scale_and_save(img=img, path=args.file)
    add_brightness_and_save(img=img, path=args.file)
