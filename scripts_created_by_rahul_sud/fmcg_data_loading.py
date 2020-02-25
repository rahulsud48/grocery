import cv2
import matplotlib.pyplot as plt
import os
import sys
import random
import PIL
import pandas as pd
import numpy as np
from tqdm import tqdm


class Process_Dataset(object):

    def __init__(self, path_img, path_metadata):
        self.path_img = path_img
        self.path_metadata = path_metadata
        img_list = os.listdir(self.path_img)
        random.shuffle(img_list)
        self.img_list = img_list
        self.df = self.preprocess_metadata(self.path_metadata,self.path_img)
        return None


    def split_list_nChunks(self, list_, file_name, nChuncks = 5):
        nChunks_list = [list_[i * nChuncks:(i + 1) * nChuncks] for i in range(int(len(list_) / nChuncks))] 
        bbox_list = []
        for chunk in nChunks_list:
            bbox_list.append(chunk + [file_name])
        return bbox_list


    def get_image_size(self, df, path_sub_list):  
        width_dict = {}
        height_dict = {}
        print("Loading Images ...")
        for filename_ in tqdm(df['filename'].unique()):
            img_ = plt.imread(os.path.join(path_sub_list,filename_))
            width_dict[filename_] = img_.shape[1]
            height_dict[filename_] = img_.shape[0]  
        df['width'] = df['filename'].map(width_dict)
        df['height'] = df['filename'].map(height_dict)
        return df


    def csv_for_tfrecords(self, df, sub_list, path_sub_list):
        df.columns = ['xmin','ymin','w','h','class','filename']
        df['xmax'] = df['xmin'].astype(int) + df['w'].astype(int)
        df['ymax'] = df['ymin'].astype(int) + df['h'].astype(int)
        df = df[df['filename'].isin(sub_list)]
        df = self.get_image_size(df, path_sub_list)
        df = df[['filename','width','height','class','xmin','ymin','xmax','ymax']]
        return df


    def preprocess_metadata(self, path_metadata, subset_list_path):
        subset_img_list = os.listdir(subset_list_path)
        df = pd.read_csv(path_metadata, sep = "delimiter", header = None)
        df_splitted = pd.DataFrame()
        for row_id in range(df.shape[0]):
            box_info = self.split_list_nChunks(df.iloc[row_id,0].split(' ')[2:], df.iloc[row_id,0].split(' ')[0])
            df_ = pd.DataFrame(np.array(box_info))
            df_splitted = df_splitted.append(df_) 
        df_splitted = self.csv_for_tfrecords(df_splitted, subset_img_list, subset_list_path)
        df_splitted['class'] = 'Product'
        return df_splitted   


    def annotate_image(self, np_image, img_name, metadata_):
        df_ = metadata_[metadata_['filename'] == img_name]
        for box_ in range(df_.shape[0]):
            x1 = int(df_.iloc[box_,:]['xmin'])
            y1 = int(df_.iloc[box_,:]['ymin'])
            x2 = int(df_.iloc[box_,:]['xmax'])
            y2 = int(df_.iloc[box_,:]['ymax'])
            cv2.rectangle(np_image, (x1, y1), (x2, y2), (0,255,0), 5)
        return np_image


    def view(self, view_box = True, zoom = 2, no_views = 10):
        fig = plt.figure()
        for img_name in self.img_list[:no_views]:
            img_ = plt.imread(os.path.join(self.path_img, img_name))
            if view_box:
                img_ = self.annotate_image(img_, img_name, self.df)
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * zoom, h * zoom)
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_aspect(1)
            ax.scatter(range(10), range(0,20,2))
            dpi = fig.get_dpi()  
            plt.imshow(img_)
            plt.show()


    def create_folder(self, folder_name):
        if folder_name not in os.listdir():
            os.mkdir(os.path.join(folder_name))
        else:
            pass


    def save_csv_for_tfrecords(self, file_name):
        self.create_folder('csv_for_tfrecords')
        self.df.to_csv(os.path.join('csv_for_tfrecords',file_name + '.csv'), index = False)
        print("Please find the file in csv_for_tfrecords folder")


if __name__ == "__main__":
    path_tr_img = 'ShelfImages/train'
    path_metadata = 'grocerydataset/annotation.txt'
    load_ = Process_Dataset(path_tr_img, path_metadata)
    load_.view(view_box = True, zoom = 2, no_views = 3)
    load_.save_csv_for_tfrecords('train')
    pass