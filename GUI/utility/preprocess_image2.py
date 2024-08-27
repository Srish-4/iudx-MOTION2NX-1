import os
import cv2
import numpy as np
import argparse

def process_img(img_name):
    base_dir = os.getenv("BASE_DIR")
    img_dir = os.path.join(base_dir, "data/ImageProvider")

    # read the image
    img_path = os.path.join(img_dir, "raw_images", img_name)

    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read image in color
    except:
        print("Unable to find the image.")
        return -1

    # Normalize the image to range [0, 1]
    img = img.astype(np.float32) / 255.0

    # Resize the image to 32x32
    img = cv2.resize(img, (32, 32))
    #new code added
    image_array = np.array(img)
    image_array = image_array.astype('float32') / 255.0
    #till here

    # Save the processed image
    #prcsd_img_dir = os.path.join(img_dir, "processed_images")
    #if not os.path.exists(prcsd_img_dir):
     #   os.mkdir(prcsd_img_dir)
    #cv2.imwrite(os.path.join(prcsd_img_dir, img_name), img * 255)

    # Flatten the image and write to CSV
    flattened_img = image_array.flatten()
    #csv_path = os.path.join(img_dir, "images", "X" + os.path.splitext(img_name)[0] + ".csv")
    #np.savetxt(csv_path, flattened_img, delimiter=",")
    csv_filename = 'X' + os.path.splitext(img_name)[0] + '.csv'
    csv_path = os.path.join(img_dir, "images", csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for value in flattened_img:
            writer.writerow([value])

    return 1

def main():
    parser = argparse.ArgumentParser(description="Process the image to 32x32x3 and create CSV file.")
    parser.add_argument("-f", "--File_Name", type=str, help="Image file name that is to be processed. Please include the extension.")
    args = parser.parse_args()

    if not args.File_Name:
        print("No image file name given")
        return -1

    process_img(args.File_Name)
    print("Image processed successfully.")
    return 1

if __name__ == "__main__":
    main()

