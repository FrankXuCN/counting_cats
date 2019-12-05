import cv2
import os
import sys

def resize_img(file, source_dir, target_dir):
    file_path = os.path.join(source_dir, file)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ',img.shape)

    width = 32
    height = 32
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    print('Resized Dimensions : ',resized.shape)

    file_path = os.path.join(target_dir, file)
    cv2.imwrite(file_path, resized)

if __name__ == '__main__':
    base_dir = "./data"
    tag = "-t"
    if tag in sys.argv:
        folder = sys.argv[sys.arg.index(tag)+1]
    else:
        folder = "Output"
    if folder not in os.listdir(base_dir):
        os.mkdir(os.path.join(base_dir, folder))
    target_dir = os.path.join(base_dir, folder)

    tag = "-s"
    if tag in sys.argv:
        folder = sys.argv[sys.argv.index(tag)+1]
        if folder in os.listdir(base_dir):
            source_dir = os.path.join(base_dir, folder)
        else:
            print("did not find {} in ./data".format(folder))
            sys.exit()
    else:
        print("-s is necessary")
        sys.exit()

    for file in os.listdir(source_dir):
        resize_img(file, source_dir, target_dir)
