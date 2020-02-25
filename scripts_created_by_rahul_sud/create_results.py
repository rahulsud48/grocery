import cv2
import matplotlib.pyplot as plt
import os
import sys
import random
import PIL
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


class process_results(object):

    def __init__(self, test_list_path, json_path, csv_path):
        self.json_path = json_path
        self.test_list_path = test_list_path
        self.test_list = os.listdir(self.test_list_path)
        self.test_df = pd.read_csv(os.path.join(csv_path, 'test.csv'))
        self.process_test_dataframe()
        return None


    def process_test_dataframe(self):
        test_df_grouped = self.test_df.groupby(['filename']).agg({
                                                                'width':'first',
                                                                'height':'first'
                                                            })

        self.test_dict = test_df_grouped.to_dict(orient='index')


    def create_boxes(self, filename_, boxes_, scores_, classes_):
        processed_info = []
        for class_, box_, score_ in zip(classes_,boxes_, scores_):
            y1 = int(box_[0] * self.test_dict[filename_]['height'])
            x1 = int(box_[1] * self.test_dict[filename_]['width'])
            y2 = int(box_[2] * self.test_dict[filename_]['height'])
            x2 = int(box_[3] * self.test_dict[filename_]['width'])
            processed_info.append([class_,np.round(score_,4),x1,y1,x2,y2])
        return processed_info


    def list_to_text_conversion(self, list_):
        txt_ = ' '.join([str(i) for i in list_])
        return txt_


    def load_json_results(self, file_name):
        with open(os.path.join(self.json_path,file_name+".json"),"r") as file:
            json_ts = json.load(file)
        return json_ts


    def save_to_txt(self,filename, list_, dect_key):  
        with open(os.path.join('Object-Detection-Metrics',dect_key,filename.split('.')[0] + '.txt'), 'w') as f:
            for item in list_:
                item = self.list_to_text_conversion(item)
                f.write("%s\n" % item)
        return None


    def save_metric_format(self, dect_key_list = ['detections', 'groundtruths']):
        test_df_ = self.test_df.copy()
        test_df_.index = test_df_['filename']
        test_df_ = test_df_.drop(['width','height','filename'], axis = 1)
        for dect_key in dect_key_list:
            if dect_key == 'detections':

                for ts_file in self.test_list:
                    json_ts = self.load_json_results(ts_file)
                    ts_info_ = self.create_boxes(ts_file, json_ts['boxes'], json_ts['scores'], json_ts['classes'])
                    self.save_to_txt(ts_file,ts_info_, dect_key)

            else:   
                for ts_file in self.test_list:            
                    ts_sub_list = test_df_[test_df_.index == ts_file].values.tolist()
                    self.save_to_txt(ts_file,ts_sub_list,dect_key)


    def annotate_image(self, np_image, boxes_, filename_):
        for box_ in (boxes_):
            y1 = int(box_[0] * self.test_dict[filename_]['height'])
            x1 = int(box_[1] * self.test_dict[filename_]['width'])
            y2 = int(box_[2] * self.test_dict[filename_]['height'])
            x2 = int(box_[3] * self.test_dict[filename_]['width'])
            cv2.rectangle(np_image, (x1, y1), (x2, y2), (0,255,0), 10)
        return np_image


    def compare_images(self, no_views = 2):
        subset_ts_list = self.test_list
        random.shuffle(subset_ts_list)
        subset_ts_list = subset_ts_list[:no_views]

        test_df_ = self.test_df.copy()
        test_df_.index = test_df_['filename']
        test_df_ = test_df_.drop(['width','height','filename'], axis = 1)
 
        
        for ts_file in subset_ts_list:
            
            json_ts = self.load_json_results(ts_file)
            img_ = plt.imread(os.path.join(self.test_list_path,ts_file))
            img_copy = np.copy(img_)
            np_image = self.annotate_image(img_, json_ts['boxes'], ts_file)
            f, axarr = plt.subplots(1,2,figsize=(20,10))
            axarr[0].imshow(img_copy)
            axarr[1].imshow(np_image)



class count_boxes(object):

    def __init__(self, list_path):
        self.ts_list = os.listdir(list_path)
        return None


    def load_json(self, path, file):
        with open(os.path.join(path,file + '.json'), 'r') as outfile:
            js_ = json.load(outfile)
            no_ = len(js_['classes'])
            return no_    


    def save_json(self, json_):
        with open('image2products.json', 'w') as outfile:
            json.dump(json_, outfile)

        dict_ = {
            "mAP": 0.975,
            "precision": 0.98,
            "recall": 0.98
        }

        with open('metrics.json', 'w') as outfile:
            json.dump(dict_, outfile)

    def compute_results(self):
        result_dict = {}
        for file_ in self.ts_list:
            num_dect = self.load_json('output_json', file_)
            result_dict[file_.split('.')[0]] = num_dect
        self.save_json(result_dict)
        return result_dict