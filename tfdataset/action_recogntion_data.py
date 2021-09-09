import os
import csv
import json
import argparse
import numpy as np

from pathlib import Path
from glob import glob
from .client import *

class ActionData:
    def __init__(self, args) -> None:
        self.object_name = args.object_name
        self.video_folder = args.video_folder
        self.output_name = args.output_name
        self.include_label = args.include_label
        self.output_folder = args.output_folder
        self.label = args.label
        self.data = None
        self.count = None

    def generate_data(self, filename, project_name='data', stride=1):
        f = open(Path(__file__).parent / "label_maps/{}.json".format(self.object_name), "r")
        label_list = json.load(f)
        label_list = label_list.fromkeys(label_list, 0)
        self.data = np.zeros((1500, len(label_list)))
        self.count = list(np.zeros((len(label_list))))

        results = run(filename, project_name, self.object_name, stride=stride)
        for image in results.image:
            self.search_object_score(image)
        self.data = self.data.flatten()
        self.data = np.insert(self.data, 0, np.array(self.count), axis=0)

        if self.include_label:
            self.data = np.concatenate((self.data, self.label))
            return self.data
        return self.data

    def search_object_score(self, image):
        index = image.index
        if index > self.data.shape[0]:
            pass
        for bbox in image.bbox:
            class_index = bbox.cls
            score = bbox.score
            self.data[index-1, class_index-1] = score
            self.count[class_index-1] += 1
            
    def generate_csv(self):
        for video_name in glob(os.path.join(self.video_folder, "*.mp4")):
            csv_data = open(os.path.join(self.output_folder, self.output_name), 'a', newline='')
            csv_line = self.generate_data(video_name)
            wr = csv.writer(csv_data)
            wr.writerow(csv_line)
            csv_data.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample Action Recognition Data Generation")

    parser.add_argument("-o",
                        "--object_name",
                        help="Object name, either glucometer, eyedrop, nasal_spray, or anything",
                        type=str, default='glucometer')
    parser.add_argument("-v",
                        "--video_folder",
                        help="Path to the video folder.",
                        type=str, default='test_videos')
    parser.add_argument("-n",
                        "--output_name",
                        help="Csv file name",
                        type=str, default='glucometer.csv')
    parser.add_argument("-i",
                        "--include_label",
                        help="Whether to include label or not",
                        type=bool, default=False)
    parser.add_argument("-f",
                        "--output_folder",
                        help="Folder to store the output csv",
                        type=str, default='output')
    parser.add_argument("-l",
                        "--label",
                        help="Label to add to the end of every line.",
                        type=str, default=None)
    

    args = parser.parse_args()
    doing_data = ActionData(args)
    doing_data.generate_csv()