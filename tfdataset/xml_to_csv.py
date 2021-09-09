from bs4 import BeautifulSoup
import argparse
import random
import os
from glob import glob


def main(args=None):
    ING_BASE_PATH = args.image_dir
    ANNOT_BASE_PATH = args.xml_dir
    CSV_SAVE_PATH = args.csv_path

    i = 0

    train_csv = os.path.sep.join([CSV_SAVE_PATH, 'train.csv'])
    test_csv = os.path.sep.join([CSV_SAVE_PATH, 'test.csv'])
    train_test_split = 0.80

    # grab all image paths then construct the training and testing split
    imagePaths = glob(os.path.join(ING_BASE_PATH, '*'))
    random.shuffle(imagePaths)
    i = int(len(imagePaths) * train_test_split)
    trainImagePaths = imagePaths[:i]
    testImagePaths = imagePaths[i:]

    # create the list of datasets to build
    dataset = [ ("train", trainImagePaths, train_csv),
                ("test", testImagePaths, test_csv)]

    # initialize the set of classes we have
    CLASSES = set()

    # loop over the datasets
    for (dType, imagePaths, outputCSV) in dataset:
        # load the contents
        print ("[INFO] creating '{}' set...".format(dType))
        print ("[INFO] {} total images in '{}' set".format(len(imagePaths), dType))

        for imagePath in imagePaths:
            # open the output CSV file
            if int(imagePaths.index(imagePath)) == 0:
                csv = open(outputCSV, "w")
                first_row = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                csv.write("{}\n".format(",".join(first_row)))
            else:
                csv = open(outputCSV, "a")

            # build the corresponding annotation path
            fname_jpg = imagePath.split(os.path.sep)[-1]
            # print(fname_jpg)
            fname_xml = "{}.xml".format(fname_jpg[:fname_jpg.rfind(".")])
            annotPath = os.path.sep.join([ANNOT_BASE_PATH, fname_xml])

            # load the contents of the annotation file and buid the soup
            if not os.path.isfile(annotPath):
                continue
            try:
                contents = open(annotPath).read()
            except:
                contents = open(annotPath, 'r', encoding='UTF-8').read()
            soup = BeautifulSoup(contents, "html.parser")

            # extract the image dimensions
            w = int(float(soup.find("width").string))
            h = int(float(soup.find("height").string))

            # loop over all object elements
            for o in soup.find_all("object"):
                #extract the label and bounding box coordinates
                label = o.find("name").string

                xMin = int(float(o.find("xmin").string))
                yMin = int(float(o.find("ymin").string))
                xMax = int(float(o.find("xmax").string))
                yMax = int(float(o.find("ymax").string))

                # truncate any bounding box coordinates that fall outside
                # the boundaries of the image
                xMin = max(0, xMin)
                yMin = max(0, yMin)
                xMax = min(w, xMax)
                yMax = min(h, yMax)

                # ignore the bounding boxes where the minimum values are larger
                # than the maximum values and vice-versa due to annotation errors
                if xMin >= xMax or yMin >= yMax:
                    continue
                elif xMax <= xMin or yMax <= yMin:
                    continue

                # write the image path, bb coordinates, label to the output CSV
                row = [fname_jpg, str(w), str(h), str(label), str(xMin), str(yMin), str(xMax),
                        str(yMax)]
                csv.write("{}\n".format(",".join(row)))

                # update the set of unique class labels
                CLASSES.add(label)

        # close the CSV file
        print(len(CLASSES), CLASSES)
        csv.close()


if __name__ == '__main__':
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-TFRecord converter")
    parser.add_argument("-x",
                        "--xml_dir",
                        help="Path to the folder where the input .xml files are stored.",
                        type=str)
    parser.add_argument("-i",
                        "--image_dir",
                        help="Path to the folder where the input image files are stored. "
                            "Defaults to the same directory as XML_DIR.",
                        type=str, default=None)
    parser.add_argument("-c",
                        "--csv_path",
                        help="Path of output .csv file. If none provided, then no file will be "
                            "written.",
                        type=str, default=None)

    args = parser.parse_args()
    main(args)