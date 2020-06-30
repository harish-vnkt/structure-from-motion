import numpy as np
import cv2
import logging


def get_keypoints_from_indices(keypoints1, index_list1, keypoints2, index_list2):
    """Filters a list of keypoints based on the indices given"""

    points1 = np.array([kp.pt for kp in keypoints1])[index_list1]
    points2 = np.array([kp.pt for kp in keypoints2])[index_list2]
    return points1, points2


def get_3D_point(u1, P1, u2, P2):
    """Solves for 3D point using homogeneous 2D points and the respective camera matrices"""

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
    """Removes outlier keypoints using the fundamental matrix"""

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
    """Calculates the reprojection error for a 3D point by projecting it back into the image plane"""
    
    reprojected_point = K @ P @ point_3D.T
    reprojected_point = cv2.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    error = np.linalg.norm(point_2D.reshape((2, 1))-reprojected_point)
    return error
