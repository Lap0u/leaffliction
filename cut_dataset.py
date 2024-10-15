import argparse
import Distribution
import balance_dataset


def cut_dataset(path, size, output):
    """Cut the dataset to a smaller size"""
    images = Distribution.fetch_images(path)
    count_dir = Distribution.count_images_per_directory(images)
    count_dir = Distribution.group_data_by_class(count_dir)
    classes = balance_dataset.flatten_dir(count_dir)
    print(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut the dataset to a smaller size")
    parser.add_argument("-d", "--dataset", help="Dataset to cut", required=True)
    parser.add_argument("-s", "--size", help="Size to cut", required=True)
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    args = parser.parse_args()
    cut_dataset(path=args.dataset, size=args.size, output=args.output)
