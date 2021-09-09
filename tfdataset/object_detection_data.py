import argparse

from .xml_to_csv import main as xml2csv
from .csv_to_tfrecords import main as csv2tfrecords

def obj_data(args):
    if args.task == 'csv':
        xml2csv(args)
    elif args.task == 'tfrecords':
        csv2tfrecords(args)
    else:
        xml2csv(args)
        csv2tfrecords(args)

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
    parser.add_argument("-d",
                        "--data_type",
                        help="data type, either train or test",
                        type=str, default=None)
    parser.add_argument("-o",
                        "--object",
                        help="object name, either eyedrop, nasalspray, blister or so on",
                        type=str, default=None)
    parser.add_argument("-t",
                        "--task",
                        help="task to do, csv, tfrecords, or all",
                        type=str, default=None)
    

    args = parser.parse_args()
    obj_data(args)