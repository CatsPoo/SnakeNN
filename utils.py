import numpy as np

def manhattan_distance(point1, point2):
    return np.abs(point2[0] - point1[0]) + np.abs(point2[1] - point1[1])
