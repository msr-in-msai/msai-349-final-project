import os
import json
import pickle
import cv2
import numpy as np

class DeleteUselessData:
    def __init__(self):
        pass
    
    def read_json_file(self, json_file_path):
        data = None
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def read_pickle_file(self, pickle_file_path):
        data = None
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def read_image_file(self, image_path):
        image = cv2.imread(image_path)
        return image
    
    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        else:
            print(f"{file_path} does not exist.")

    def get_objects_from_seg_image(self, seg_image, seg_labels, occluder_name):
        obj_types = []
        num_rows = seg_image.shape[0]
        num_columns = seg_image.shape[1]
        for i in range(num_rows):
            for j in range(num_columns):
                rgb = seg_image[i, j]
                if(rgb[0]!=0 and rgb[1]!=0 and rgb[2]!=0):
                    argb = '('+str(rgb[2])+', '+str(rgb[1])+', '+str(rgb[0])+', 255)'
                    obj_type = seg_labels[argb]['class']
                    if obj_type not in obj_types:
                        obj_types.append(obj_type)
        return obj_types

    def get_objects_from_seg_image_np(self, seg_image, seg_labels, occluder_name):
        obj_types = []
        # Filter out black pixels (which represent the background)
        non_black_pixels = seg_image[(seg_image[:, :, 0] != 0) | (seg_image[:, :, 1] != 0) | (seg_image[:, :, 2] != 0)]
        # Create an array of ARGB strings for non-black pixels
        argb_strings = ['({}, {}, {}, 255)'.format(rgb[2], rgb[1], rgb[0]) for rgb in non_black_pixels]
        # Get unique ARGB values (only unique object colors)
        unique_argb_strings = np.unique(argb_strings)
        # Map unique ARGB values to object types
        for argb in unique_argb_strings:
            obj_type = seg_labels.get(argb, {}).get('class')
            if obj_type and obj_type not in obj_types:
                obj_types.append(obj_type)
        return obj_types
    
    def check_if_should_delete_the_data(self, obj_types, occluder_name, fall_object_names=None):
        if(len(obj_types)==0):
            return True
        else:
            delete = True
            for obj in obj_types:
                if(obj != occluder_name):
                    delete = False
            return delete

    def read_objects(self, dataset_path):
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if(path[-4:] ==".png" and path[-30:-8]=="semantic_segmentation_"):
                    parent_folder_directory = path[:-30]
                    data_id = path[-8:-4]

                    segmantic_segmentation_path = path
                    segmantic_segmentation_labels_path = parent_folder_directory + "semantic_segmentation_labels_"+data_id+".json"
                    rgb_image_path = parent_folder_directory+"rgb_"+data_id+".png"
                    depth_image_path = parent_folder_directory+"distance_to_image_plane_"+data_id+".npy"
                    camera_params_path = parent_folder_directory+"camera_params_"+data_id+".json"
                    occluder_name_path = parent_folder_directory+"occluder.pickle"
                    fall_objects_path = parent_folder_directory+"fall_objects.pickle"                    

                    occluder_name = self.read_pickle_file(occluder_name_path)[0]
                    fall_objects = self.read_pickle_file(fall_objects_path)
                    seg_labels = self.read_json_file(segmantic_segmentation_labels_path)

                    seg_image = np.asanyarray(self.read_image_file(segmantic_segmentation_path))
                    obj_types = self.get_objects_from_seg_image_np(seg_image, seg_labels, occluder_name)

                    delete = self.check_if_should_delete_the_data(obj_types, occluder_name, fall_objects)

                    if(delete == True):
                        # print(segmantic_segmentation_path)
                        self.delete_file(segmantic_segmentation_path)
                        self.delete_file(segmantic_segmentation_labels_path)
                        self.delete_file(rgb_image_path)
                        self.delete_file(depth_image_path)
                        self.delete_file(camera_params_path)
                        pass
                    # break
            # break


dud = DeleteUselessData()
dud.read_objects("/home/zhengxiao-han/Downloads/Datasets/msai_dataset/train/")
# start with 770