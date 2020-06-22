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
