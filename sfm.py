import numpy as np
import os
from utils import *
import open3d as o3d


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
        self.done = []

        for view in self.views:
            self.names.append(view.name)

        if not os.path.exists(self.views[0].root_path + '/points'):
            os.makedirs(self.views[0].root_path + '/points')

        self.results_path = self.views[0].root_path + '/points/'

    def get_index_of_view(self, view):

        return self.names.index(view.name)

    def remove_mapped_points(self, match_object, image_idx):

        inliers1 = []
        inliers2 = []

        for i in range(len(match_object.inliers1)):
            if (image_idx, match_object.inliers1[i]) not in self.point_map:
                inliers1.append(match_object.inliers1[i])
                inliers2.append(match_object.inliers2[i])

        match_object.inliers1 = inliers1
        match_object.inliers2 = inliers2

    def compute_pose(self, view1, view2=None, is_baseline=False):

        if is_baseline and view2:

            match_object = self.matches[(view1.name, view2.name)]
            F = remove_outliers_using_F(view1, view2, match_object)

            E = self.K.T @ F @ self.K
            view1.R = np.eye(3, 3)
            view2.R, view2.t = get_camera_from_E(E)
            self.triangulate(view1, view2)
            self.done.append(view1)
            self.done.append(view2)

        else:

            view1.R, view1.t = self.compute_pose_PNP(view1)

            for i, old_view in enumerate(self.done):

                match_object = self.matches[(old_view.name, view1.name)]
                _ = remove_outliers_using_F(old_view, view1, match_object)
                self.remove_mapped_points(match_object, i)
                self.triangulate(old_view, view1)

            self.done.append(view1)

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

    def compute_pose_PNP(self, view):

        if view.feature_type in ['sift', 'surf']:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        old_descriptors = []
        for old_view in self.done:
            old_descriptors.append(old_view.descriptors)

        matcher.add(old_descriptors)
        matcher.train()
        matches = matcher.match(queryDescriptors=view.descriptors)
        points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))

        for match in matches:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = match.imgIdx, match.queryIdx, match.trainIdx

            if (old_image_idx, old_image_kp_idx) in self.point_map:
                point_2D = np.array(view.keypoints[new_image_kp_idx].pt).T.reshape((1, 2))
                points_2D = np.concatenate((points_2D, point_2D), axis=0)
                point_3D = self.points_3D[self.point_map[(old_image_idx, old_image_kp_idx)], :].T.reshape((1, 3))
                points_3D = np.concatenate((points_3D, point_3D), axis=0)

        _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], points_2D[:, np.newaxis], self.K, None)
        R, _ = cv2.Rodrigues(R)
        return R, t

    def plot_points(self):

        number = len(self.done)
        filename = self.results_path + str(number) + '_images.ply'
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_3D)
        o3d.io.write_point_cloud(filename, pcd)

    def reconstruct(self):

        baseline_view1, baseline_view2 = self.views[0], self.views[1]
        self.compute_pose(view1=baseline_view1, view2=baseline_view2, is_baseline=True)
        self.plot_points()

        for i in range(2, len(self.views)):

            self.compute_pose(view1=self.views[i])
            self.plot_points()