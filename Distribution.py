import argparse
import os

MAX_DEPTH = 4
MAIN_CLASSES = ["Apple", "Grape"]


def group_data_by_class(count_dir):
    """Group data by class"""
    data = {}
    for main_class in MAIN_CLASSES:
        data[main_class] = {}
    for dir in count_dir:
        classes = dir.split("_")
        if classes[0] in MAIN_CLASSES:
            data[classes[0]][dir] = count_dir[dir]
    return data


def plot_sub_data(data, title):
    """Plot the distribution of images per class"""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"{title} Distribution")
    axs[0].bar(data.keys(), data.values())
    axs[1].pie(data.values(), labels=data.keys(), autopct="%1.1f%%")
    plt.show()


def plot_distribution_per_class(data):
    """Plot the distribution of images per class"""
    for key in data:
        plot_sub_data(data[key], key)


def fetch_directory(directory, files, depth=0):
    """Fetch all files and directoires in a given directory"""
    if depth > MAX_DEPTH:
        return

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            fetch_directory(path, files, depth + 1)
        else:
            files.append(path)
    return files


def remove_higher_directories(count_dir):
    """Keep only the last subpath as the directory name"""
    for directory in list(count_dir.keys()):
        subpath = directory.split("/")[-1]
        if subpath != directory:
            count_dir[subpath] = count_dir.pop(directory)
    return count_dir


def count_images_per_directory(images):
    """Count number of images in each directory and return a dictionary"""
    count_dir = {}
    for image in images:
        directory = os.path.dirname(image)
        if directory in count_dir:
            count_dir[directory] += 1
        else:
            count_dir[directory] = 1
    count_dir = remove_higher_directories(count_dir)
    return count_dir


def fetch_images(directory):
    """Fetch all images in a given directory"""
    files = []
    fetch_directory(directory, files, depth=0)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore and analyze data in a given directory"
    )
    parser.add_argument("-d", "--directory", help="Directory to explore", required=True)
    args = parser.parse_args()
    images = fetch_images(args.directory)
    count_dir = count_images_per_directory(images)
    count_dir = group_data_by_class(count_dir)
    plot_distribution_per_class(count_dir)
