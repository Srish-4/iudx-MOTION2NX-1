import os
import csv
import cv2
import numpy as np
import argparse
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = [150, 150]


def process_img(img_name):
    base_dir = os.getenv("BASE_DIR")
    img_dir = os.path.join(base_dir, "data/ImageProvider")

    # read the image
    img_path = os.path.join(img_dir, "raw_images", img_name)
    print(img_path)
    # try:
    #   img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read image in color
    # except:
    #   print("Unable to find the image.")
    #  return -1
    img = tf.io.read_file(img_path)

    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the desired size
    processed_img = tf.image.resize(img, IMAGE_SIZE)

    image_array = processed_img.numpy()

    print(image_array.shape)
    print(image_array.dtype)

    image_shape = image_array.shape

    arr = []
    for p in range(image_shape[2]):  # channels
        for q in range(image_shape[0]):  # rows
            for r in range(image_shape[1]):  # cols
                arr.append(image_array[q][r][p])

    print(len(arr))

    # Save the processed image
    # prcsd_img_dir = os.path.join(img_dir, "processed_images")
    # if not os.path.exists(prcsd_img_dir):
    #   os.mkdir(prcsd_img_dir)
    # cv2.imwrite(os.path.join(prcsd_img_dir, img_name), img * 255)

    # Flatten the image and write to CSV
    # flattened_img = image_array.flatten()
    csv_path = os.path.join(img_dir, "images", "X" +
                            os.path.splitext(img_name)[0] + ".csv")
    print(csv_path)
    # np.savetxt(csv_path, flattened_img, delimiter=",")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for value in arr:
            writer.writerow([value])

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Process the image to 3*150*150 and create CSV file.")
    parser.add_argument("-f", "--File_Name", type=str,
                        help="Image file name that is to be processed. Please include the extension.")
    args = parser.parse_args()

    if not args.File_Name:
        print("No image file name given")
        return -1

    process_img(args.File_Name)
    print("Image processed successfully.")
    return 1


if __name__ == "__main__":
    main()
