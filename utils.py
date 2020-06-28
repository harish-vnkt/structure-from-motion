import numpy as np
import cv2


def get_keypoints_from_indices(keypoints1, index_list1, keypoints2, index_list2):

    points1 = np.array([kp.pt for kp in keypoints1])[index_list1]
    points2 = np.array([kp.pt for kp in keypoints2])[index_list2]
    return points1, points2


def get_camera_from_E(E):

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, _, vt = np.linalg.svd(E)
    R = u @ W @ vt
    t = u[:, -1].reshape((3, 1))
    return R, t


def get_3D_point(u1, P1, u2, P2):

    A = np.array([[u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1], u1[0] * P1[2, 2] - P1[0, 2]],
                  [u1[1] * P1[2, 0] - P1[1, 0], u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2]],
                  [u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1], u2[0] * P2[2, 2] - P2[0, 2]],
                  [u2[1] * P2[2, 0] - P2[1, 0], u2[1] * P2[2, 1] - P2[1, 1], u2[1] * P2[2, 2] - P2[1, 2]]])

    B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                  -(u1[1] * P1[2, 3] - P1[1, 3]),
                  -(u2[0] * P2[2, 3] - P2[0, 3]),
                  -(u2[1] * P2[2, 3] - P2[1, 3])])

    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X[1]


def remove_outliers_using_F(view1, view2, match_object):

    pixel_points1, pixel_points2 = get_keypoints_from_indices(keypoints1=view1.keypoints,
                                                              keypoints2=view2.keypoints,
                                                              index_list1=match_object.indices1,
                                                              index_list2=match_object.indices2)
    F, mask = cv2.findFundamentalMat(pixel_points1, pixel_points2, method=cv2.FM_RANSAC,
                                     ransacReprojThreshold=0.1, confidence=0.99)
    mask = mask.astype(bool).flatten()
    match_object.inliers1 = np.array(match_object.indices1)[mask]
    match_object.inliers2 = np.array(match_object.indices2)[mask]

    return F


def calculate_reprojection_error(point_3D, point_2D, K, P):

    reprojected_point = K @ P @ point_3D.T
    reprojected_point = cv2.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    error = np.linalg.norm(point_2D.reshape((2, 1))-reprojected_point)
    return error
