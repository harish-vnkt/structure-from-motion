from utils import *
import logging


class Baseline:
    """Represents the functions that compute the baseline pose from the initial images of a reconstruction"""

    def __init__(self, view1, view2, match_object):

        self.view1 = view1  # first view
        self.view1.R = np.eye(3, 3)  # identity rotation since the first view is said to be at the origin
        self.view2 = view2  # second view
        self.match_object = match_object  # match object between first and second view

    def get_pose(self, K):
        """Computes and returns the rotation and translation components for the second view"""

        F = remove_outliers_using_F(self.view1, self.view2, self.match_object)
        E = K.T @ F @ K  # compute the essential matrix from the fundamental matrix
        logging.info("Computed essential matrix")
        logging.info("Choosing correct pose out of 4 solutions")

        return self.check_pose(E, K)

    def check_pose(self, E, K):
        """Retrieves the rotation and translation components from the essential matrix by decomposing it and verifying the validity of the 4 possible solutions"""

        R1, R2, t1, t2 = get_camera_from_E(E)  # decompose E
        if not check_determinant(R1):
            R1, R2, t1, t2 = get_camera_from_E(-E)  # change sign of E if R1 fails the determinant test

        # solution 1
        reprojection_error, points_3D = self.triangulate(K, R1, t1)
        # check if reprojection is not faulty and if the points are correctly triangulated in the front of the camera
        if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R1, t1))):

            # solution 2
            reprojection_error, points_3D = self.triangulate(K, R1, t2)
            if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R1, t2))):

                # solution 3
                reprojection_error, points_3D = self.triangulate(K, R2, t1)
                if reprojection_error > 100.0 or not check_triangulation(points_3D, np.hstack((R2, t1))):

                    # solution 4
                    return R2, t2

                else:
                    return R2, t1

            else:
                return R1, t2

        else:
            return R1, t1

    def triangulate(self, K, R, t):
        """Triangulate points between the baseline views and calculates the mean reprojection error of the triangulation"""

        K_inv = np.linalg.inv(K)
        P1 = np.hstack((self.view1.R, self.view1.t))
        P2 = np.hstack((R, t))

        # only reconstructs the inlier points filtered using the fundamental matrix
        pixel_points1, pixel_points2 = get_keypoints_from_indices(keypoints1=self.view1.keypoints,
                                                                  keypoints2=self.view2.keypoints,
                                                                  index_list1=self.match_object.inliers1,
                                                                  index_list2=self.match_object.inliers2)

        # convert 2D pixel points to homogeneous coordinates
        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]

        reprojection_error = []

        points_3D = np.zeros((0, 3))  # stores the triangulated points

        for i in range(len(pixel_points1)):
            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            # convert homogeneous 2D points to normalized device coordinates
            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            # calculate 3D point
            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)

            # calculate reprojection error
            error = calculate_reprojection_error(point_3D, u2[0:2], K, R, t)
            reprojection_error.append(error)

            # append point
            points_3D = np.concatenate((points_3D, point_3D.T), axis=0)

        return np.mean(reprojection_error), points_3D
