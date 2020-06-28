from utils import *


class Baseline:

    def __init__(self, view1, view2, match_object):

        self.view1 = view1
        self.view1.R = np.eye(3, 3)
        self.view2 = view2
        self.match_object = match_object

    def get_pose(self, K):

        F = remove_outliers_using_F(self.view1, self.view2, self.match_object)
        E = K.T @ F @ K

        return self.check_pose(E, K)

    def check_pose(self, E, K):

        R1, R2, t1, t2 = Baseline.get_camera_from_E(E)
        if not Baseline.check_determinant(R1):
            R1, R2, t1, t2 = Baseline.get_camera_from_E(-E)

        reprojection_error, points_3D = self.triangulate(K, R1, t1)

        if reprojection_error > 100.0 or not Baseline.check_triangulation(points_3D, np.hstack((R1, t1))):
            reprojection_error, points_3D = self.triangulate(K, R1, t2)

            if reprojection_error > 100.0 or not Baseline.check_triangulation(points_3D, np.hstack((R1, t2))):
                reprojection_error, points_3D = self.triangulate(K, R2, t1)

                if reprojection_error > 100.0 or not Baseline.check_triangulation(points_3D, np.hstack((R2, t1))):
                    return R2, t2

                else:
                    return R2, t1

            else:
                return R1, t2

        else:
            return R1, t1

    def triangulate(self, K, R, t):

        K_inv = np.linalg.inv(K)
        P1 = np.hstack((self.view1.R, self.view1.t))
        P2 = np.hstack((R, t))

        pixel_points1, pixel_points2 = get_keypoints_from_indices(keypoints1=self.view1.keypoints,
                                                                  keypoints2=self.view2.keypoints,
                                                                  index_list1=self.match_object.inliers1,
                                                                  index_list2=self.match_object.inliers2)

        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]
        reprojection_error = []
        points_3D = np.zeros((0, 3))

        for i in range(len(pixel_points1)):
            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)
            point_3D_homogeneous = cv2.convertPointsToHomogeneous(point_3D.T)[:, 0, :]
            error = calculate_reprojection_error(point_3D_homogeneous, u2[0:2], K, P2)
            reprojection_error.append(error)
            points_3D = np.concatenate((points_3D, point_3D.T), axis=0)

        return np.mean(reprojection_error), points_3D

    @staticmethod
    def get_camera_from_E(E):

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        W_t = W.T
        u, w, vt = np.linalg.svd(E)

        R1 = u @ W @ vt
        R2 = u @ W_t @ vt
        t1 = u[:, -1].reshape((3, 1))
        t2 = - t1
        return R1, R2, t1, t2

    @staticmethod
    def check_determinant(R):

        if np.linalg.det(R) + 1.0 < 1e-9:
            return False
        else:
            return True

    @staticmethod
    def check_triangulation(points, P):

        P = np.vstack((P, np.array([0, 0, 0, 1])))
        reprojected_points = cv2.perspectiveTransform(points, P)
        z = reprojected_points[:, -1]
        if (np.sum(z > 0)/z.shape[0]) < 0.75:
            return False
        else:
            return True
