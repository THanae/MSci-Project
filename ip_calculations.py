import numpy as np


def line_line_intersection(vector1, point1, vector2, point2):
    intersections = []
    for i in range(len(vector1)):
        unit_vector1 = np.array(vector1.loc[i]) / np.linalg.norm(vector1.loc[i])
        unit_vector2 = np.array(vector2.loc[i]) / np.linalg.norm(vector2.loc[i])
        x_s = []
        combinations = [[0, 1], [0, 2], [1, 2]]
        for comb in combinations:
            a = np.array([[unit_vector1[k], - unit_vector2[k]] for k in comb])
            b = np.array([point2[i][k] - point1[i][k] for k in comb])
            x = np.linalg.solve(a, b)
            x_s.append(x)
        a = np.mean(x_s, axis=0)[0]
        # b = np.mean(x_s[1])
        intersections.append(a * unit_vector1 + point1[i])
    return intersections


def line_point_distance(vector, vector_point, point, direction=None):
    """
    Finds the distance between a series of lines and a series of points
    :param vector: series of vectors defining lines
    :param vector_point: series of point defining lines
    :param point: series of points to calculate distances to
    :return:
    """
    distances = []
    for i in range(len(vector)):
        unit_vector = np.array(vector.loc[i]) / np.linalg.norm(vector.loc[i])
        component_perp_to_line = (np.array(vector_point.loc[i]) - np.array(point.loc[i])) - (
                np.dot((np.array(vector_point.loc[i]) - np.array(point.loc[i])), np.array(unit_vector)) * np.array(
            unit_vector))
        if direction is not None:
            distance = np.linalg.norm(component_perp_to_line) * np.sign(np.dot(component_perp_to_line, direction[i]))
        else:
            distance = np.linalg.norm(component_perp_to_line)
        distances.append(distance)
    return distances
