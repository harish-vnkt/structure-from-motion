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
        self.points_3D = np.zeros((0, 3))
        self.point_counter = 0
        self.point_map = {}

        for view in self.views:
            self.names.append(view.name)

    def get_index_of_view(self, view):

        return self.names.index(view.name)

    def compute_pose(self, view1, view2, is_baseline=False):

        match_object = self.matches[(view1.name, view2.name)]
        pixel_points1, pixel_points2 = get_keypoints_from_indices(keypoints1=view1.keypoints, keypoints2=view2.keypoints, index_list1=match_object.indices1,
                                                                  index_list2=match_object.indices2)
        F, mask = cv2.findFundamentalMat(pixel_points1, pixel_points2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.1, confidence=0.99)
        mask = mask.astype(bool).flatten()
        match_object.inliers1 = np.array(match_object.indices1)[mask]
        match_object.inliers2 = np.array(match_object.indices2)[mask]

        if is_baseline:
            E = self.K.T @ F @ self.K
            view1.R = np.eye(3, 3)
            view2.R, view2.t = get_camera_from_E(E)
            self.triangulate(view1, view2)

    def triangulate(self, view1, view2):

        K_inv = np.linalg.inv(self.K)
        P1 = np.hstack((view1.R, view1.t))
        P2 = np.hstack((view2.R, view2.t))

        match_object = self.matches[(view1.name, view2.name)]
        pixel_points1, pixel_points2 = get_keypoints_from_indices(keypoints1=view1.keypoints,
                                                                  keypoints2=view2.keypoints,
                                                                  index_list1=match_object.inliers1,
                                                                  index_list2=match_object.inliers2)
        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]

        for i in range(len(pixel_points1)):

            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)
            self.points_3D = np.concatenate((self.points_3D, point_3D.T), axis=0)

            self.point_map[(self.get_index_of_view(view1), match_object.inliers1[i])] = self.point_counter
            self.point_map[(self.get_index_of_view(view2), match_object.inliers2[i])] = self.point_counter
            self.point_counter += 1

    def reconstruct(self):

        baseline_view1, baseline_view2 = self.views[0], self.views[1]
        self.compute_pose(view1=baseline_view1, view2=baseline_view2, is_baseline=True)
