import numpy as np
import cv2
from utils import *


class SFM:

    def __init__(self, views, matches, K):

        self.views = views
        self.matches = matches
        self.names = []
        self.done = []
        self.K = K

        for view in self.views:
            self.names.append(view.name)

    def compute_pose(self, view1, view2, is_baseline=False):

        match_object = self.matches[(view1.name, view2.name)]
        pixel_points2, pixel_points1 = get_keypoints_from_indices(keypoints1=view1.keypoints, keypoints2=view2.keypoints, index_list1=match_object.indices1,
                                                                  index_list2=match_object.indices2)
        F, mask = cv2.findFundamentalMat(pixel_points1, pixel_points2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.1, confidence=0.99)
        mask = mask.astype(bool).flatten()
        view1.point_map[np.array(match_object.indices1)[mask]] = True
        view2.point_map[np.array(match_object.indices2)[mask]] = True

        if is_baseline:
            E = self.K.T @ F @ self.K
            view1.R = np.eye(3, 3)
            view2.R, view2.t = get_camera_from_E(E)


    def reconstruct(self):

        baseline_view1, baseline_view2 = self.views[0], self.views[1]
        self.compute_pose(view1=baseline_view1, view2=baseline_view2, is_baseline=True)
