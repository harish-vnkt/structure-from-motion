import os
import pickle
import cv2


class Match:

    def __init__(self, view1, view2, match_path):

        self.indices1 = self.indices2 = self.distances = []
        self.image_name1 = view1.name
        self.image_name2 = view2.name
        self.root_path = view1.root_path

        if view1.feature_type in ['sift', 'surf']:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if not match_path:
            self.get_matches(view1, view2)
        else:
            self.read_matches()

    def get_matches(self, view1, view2):

        matches = self.matcher.match(view1.descriptors, view2.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        for i in range(len(matches)):
            self.indices1.append(matches[i].queryIdx)
            self.indices2.append(matches[i].trainIdx)
            self.distances.append(matches[i].distance)

        self.write_matches()

    def write_matches(self):

        if not os.path.exists(self.root_path + '/matches'):
            os.makedirs(self.root_path + '/matches')

        temp_array = []
        for i in range(len(self.indices1)):
            temp = (self.distances[i], self.indices1[i], self.indices2[i])
            temp_array.append(temp)

        matches_file = open(self.root_path + '/matches/' + self.image_name1 + '_' + self.image_name2 + '.pkl', 'wb')
        pickle.dump(temp_array, matches_file)
        matches_file.close()

    def read_matches(self):

        matches = pickle.load(
            open(
                self.root_path + '/matches/' + self.image_name1 + '_' + self.image_name2 + '.pkl',
                "rb"
            )
        )

        for point in matches:
            self.distances.append(point[0])
            self.indices1.append(point[1])
            self.indices2.append(point[2])


def create_matches(views):

    match_path = False

    root_path = views[0].root_path

    if os.path.exists(root_path + '/matches'):
        match_path = True

    matches = {}
    for i in range(0, len(views) - 1):
        for j in range(i+1, len(views)):
            matches[(views[i].name, views[j].name)] = Match(views[i], views[j], match_path)

    return matches
