import numpy as np
from typing import List


def find_inverse(a: np.array):
    """
    Finds the inverse of a three by three matrix
    :param a: matrix (array of arrays)
    :return:
    """
    inside_inverse = [[a[1][1] * a[2][2] - a[2][1] * a[1][2], a[0][2] * a[2][1] * a[2][2] * a[0][1],
                       a[0][1] * a[1][2] - a[1][1] * a[0][2]],
                      [a[1][2] * a[2][0] - a[2][2] * a[1][0], a[0][0] * a[2][2] - a[2][0] * a[0][2],
                       a[0][2] * a[1][0] - a[1][2] * a[0][0]],
                      [a[1][0] * a[2][1] - a[2][0] * a[1][1], a[0][1] * a[2][0] - a[2][1] * a[0][0],
                       a[0][0] * a[1][1] - a[1][0] * a[0][1]]]
    return 1 / find_determinant(a) * np.array(inside_inverse)


def find_determinant(a):
    """
    Finds the determinant of a three by three matrix
    :param a: matrix (array of arrays, or array of rows)
    :return:
    """
    positive_side = a[0][0] * a[1][1] * a[2][2] + a[1][0] * a[2][1] * a[0][2] + a[2][0] * a[0][1] * a[1][2]
    negative_side = a[1][1] * a[2][0] * a[0][2] + a[2][2] * a[1][0] * a[0][1] + a[0][0] * a[1][2] * a[2][1]
    determinant = positive_side - negative_side
    return determinant


def find_determinant_of_dir_matrix(vector_plane_lb: List, momentum_pkmu: List, momentum_tauMu: List):
    determinant = -vector_plane_lb[0] * momentum_pkmu[1] * momentum_tauMu[2] - momentum_tauMu[0] * \
                  vector_plane_lb[1] * momentum_pkmu[2] - momentum_pkmu[0] * momentum_tauMu[1] * \
                  vector_plane_lb[2] + momentum_tauMu[0] * momentum_pkmu[1] * vector_plane_lb[2] + \
                  vector_plane_lb[0] * momentum_tauMu[1] * momentum_pkmu[2] + momentum_pkmu[0] * \
                  vector_plane_lb[1] * momentum_tauMu[2]
    return determinant


def find_inverse_of_dir_matrix(vector_plane_lb: List, momentum_pkmu: List, momentum_tauMu: List, determinant: float):
    inside_inverse = [[-momentum_tauMu[2] * momentum_pkmu[1] + momentum_tauMu[1] * momentum_pkmu[2],
                       -momentum_tauMu[0] * momentum_pkmu[2] + momentum_tauMu[2] * momentum_pkmu[0],
                       -momentum_tauMu[1] * momentum_pkmu[0] + momentum_tauMu[0] * momentum_pkmu[1]],
                      [-momentum_tauMu[1] * vector_plane_lb[2] + momentum_tauMu[2] * vector_plane_lb[1],
                       -momentum_tauMu[2] * vector_plane_lb[0] + momentum_tauMu[0] * vector_plane_lb[2],
                       -momentum_tauMu[0] * vector_plane_lb[1] + momentum_tauMu[1] * vector_plane_lb[0]],
                      [vector_plane_lb[1] * momentum_pkmu[2] - vector_plane_lb[2] * momentum_pkmu[1],
                       vector_plane_lb[2] * momentum_pkmu[0] - vector_plane_lb[0] * momentum_pkmu[2],
                       vector_plane_lb[0] * momentum_pkmu[1] - vector_plane_lb[1] * momentum_pkmu[0]]]
    return 1/determinant * np.array(inside_inverse)
