import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import json
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class DetectionObj(object):


    def __init__(self):
        self.CURRENT_PATH = os.getcwd()
        self.TARGET_PATH = self.CURRENT_PATH
        self.THRESHOLD = 0.25
        self.CKPT_FILE = '/home/rahul/Documents/FMCG_project/detection_ssd/workspace/training_demo/trained-inference-graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb'
        try:
            self.DETECTION_GRAPH = self.load_frozen_model()
        except:
            print("Model Not Found ...")
        self.NUM_CLASSES = 1
        path_to_labels = os.path.join(self.CURRENT_PATH,'annotations','label_map.pbtxt')
        label_mapping = label_map_util.load_labelmap(path_to_labels)
        extracted_categories = label_map_util.convert_label_map_to_categories(label_mapping,max_num_classes=self.NUM_CLASSES,use_display_name=True)
        self.LABELS= {item['id']:item['name'] for item in extracted_categories}
        self.CATEGORY_INDEX = label_map_util.create_category_index(extracted_categories)
        self.TF_SESSION = tf.Session(graph=self.DETECTION_GRAPH)
        return None


    def load_frozen_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.CKPT_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph




    def load_image_from_disk(self, image_path):
        return Image.open(image_path)

    
    def load_image_into_numpy_array(self, image):
        try:
            im_width, im_height = image.size
            return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)
        except:
            return image

    
    
    def detect(self, images, annotate_on_image = True):
        if type(images) is not list:
            images = [images]
        results = list()
        for image in images:
            image_np = self.load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.DETECTION_GRAPH.get_tensor_by_name('image_tensor:0')
            boxes = self.DETECTION_GRAPH.get_tensor_by_name('detection_boxes:0')
            scores = self.DETECTION_GRAPH.get_tensor_by_name('detection_scores:0')
            classes = self.DETECTION_GRAPH.get_tensor_by_name('detection_classes:0')
            num_detections = self.DETECTION_GRAPH.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.TF_SESSION.run([boxes, scores, classes, num_detections], feed_dict = {image_tensor:image_np_expanded})
        if annotate_on_image:
            new_image = self.detection_on_image(image_np, boxes, scores, classes)
            results.append((new_image, boxes,scores, classes, num_detections))
        else:
            results.append((new_image, boxes,scores, classes, num_detections))
        return results


    def detection_on_image(self, image_np, boxes, scores, classes):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.CATEGORY_INDEX,
            use_normalized_coordinates=True,
            line_thickness=8
        )
        return image_np

    def visualize_image(self, image_np, target_folder, file_name):
        RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(target_folder, file_name),RGB_img)


    def serialize_annotation(self,boxes,scores,classes, file_path):
        threshold = self.THRESHOLD
        valid = [position for position, score in enumerate(scores[0]) if score > threshold]
        if len(valid) > 0:
            valid_score = scores[0][valid].tolist()
            valid_boxes = boxes[0][valid].tolist()
            valid_class = [self.LABELS[int(a_class)] for a_class in classes[0][valid]]
            with open(file_path, 'w') as outfile:
                json_data = {'classes':valid_class,
                            'boxes':valid_boxes,
                            'scores':valid_score
                            }
                json.dump(json_data,outfile)

    def save_num_detections(self, detection_dict):
        with open("total_detections.json","w") as outfile:
            json.dump(detection_dict, outfile)


    def create_folder(self, folder_name):
        if folder_name not in os.listdir():
            os.mkdir(os.path.join(folder_name))
        else:
            pass


    def file_pipeline(self, images_list, path, target_folder_json, target_folder_image):
        self.create_folder(target_folder_json)
        self.create_folder(target_folder_image)
        detect_dict ={}
        for filename in tqdm(images_list):
            single_image = self.load_image_from_disk(os.path.join(path,filename))
            
            for new_image, boxes, scores, classes, num_detection in self.detect(single_image):
                self.serialize_annotation(boxes, scores, classes, file_path = os.path.join(target_folder_json,filename + ".json"))
                self.visualize_image(new_image, target_folder_image, filename)
                detect_dict[filename] = int(num_detection)
        self.save_num_detections(detect_dict)



