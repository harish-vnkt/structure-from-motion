import os
import sys
import pickle
import cv2
import numpy as np
import glob


class View:

    def __init__(self, image_path, root_path, feature_path, feature_type='sift'):

        self.name = image_path[image_path.rfind('/') + 1:-4]
        self.image = cv2.imread(image_path)
        self.keypoints = []
        self.descriptors = []
        self.feature_type = feature_type
        self.root_path = root_path

        if not feature_path:
            self.extract_features()
        else:
            self.read_features()

    def extract_features(self):

        if self.feature_type == 'sift':
            detector = cv2.xfeatures2d.SIFT_create()
        elif self.feature_type == 'surf':
            detector = cv2.xfeatures2d.SURF_create()
        elif self.feature_type == 'orb':
            detector = cv2.ORB_create(nfeatures=1500)
        else:
            print('Admitted values for the feature detector are: sift, surf or orb ')
            sys.exit(0)

        self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)

        self.write_features()

    def read_features(self):

        features = pickle.load(open(self.root_path + '/features/' + self.name + '.pkl', "rb"))

        keypoints = []
        descriptors = []

        for point in features:
            keypoint = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                    _response=point[3], _octave=point[4], _class_id=point[5])
            descriptor = point[6]
            keypoints.append(keypoint)
            descriptors.append(descriptor)

        self.keypoints = keypoints
        self.descriptors = np.array(descriptors)

    def write_features(self):

        if not os.path.exists(self.root_path + '/features'):
            os.makedirs(self.root_path + '/features')

        temp_array = []
        for idx, point in enumerate(self.keypoints):
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id,
                    self.descriptors[idx])
            temp_array.append(temp)

        features_file = open(self.root_path + '/features/' + self.name + '.pkl', 'wb')
        pickle.dump(temp_array, features_file)
        features_file.close()


def create_views(root_path, image_format='jpg'):

    feature_path = False

    if os.path.exists(root_path + '/features'):
        feature_path = True

    image_names = sorted(glob.glob(root_path + '/images/*.' + image_format))

    views = []
    for image_name in image_names:
        views.append(View(image_name, root_path, feature_path=feature_path))

    return views
