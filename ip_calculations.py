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


def return_all_ip(data_frame):
    data_frame['vector_proton'] = data_frame[['proton_PX', 'proton_PY', 'proton_PZ']].values.tolist()
    data_frame['point_proton'] = data_frame[['proton_REFPX', 'proton_REFPY', 'proton_REFPZ']].values.tolist()
    data_frame['vector_kaon'] = data_frame[['Kminus_PX', 'Kminus_PY', 'Kminus_PZ']].values.tolist()
    data_frame['point_kaon'] = data_frame[['Kminus_REFPX', 'Kminus_REFPY', 'Kminus_REFPZ']].values.tolist()
    data_frame['vector_mu1'] = data_frame[['mu1_PX', 'mu1_PY', 'mu1_PZ']].values.tolist()
    data_frame['point_mu1'] = data_frame[['mu1_REFPX', 'mu1_REFPY', 'mu1_REFPZ']].values.tolist()
    data_frame['vector_tauMu'] = data_frame[['tauMu_PX', 'tauMu_PY', 'tauMu_PZ']].values.tolist()
    data_frame['point_tauMu'] = data_frame[['tauMu_REFPX', 'tauMu_REFPY', 'tauMu_REFPZ']].values.tolist()
    data_frame['pk_endvertex'] = line_line_intersection(data_frame['vector_proton'], data_frame['point_proton'],
                                                        data_frame['vector_kaon'], data_frame['point_kaon'])
    data_frame['pmu1_endvertex'] = line_line_intersection(data_frame['vector_proton'], data_frame['point_proton'],
                                                          data_frame['vector_mu1'], data_frame['point_mu1'])
    data_frame['ptauMu_endvertex'] = line_line_intersection(data_frame['vector_proton'], data_frame['point_proton'],
                                                            data_frame['vector_tauMu'], data_frame['point_tauMu'])
    data_frame['kmu1_endvertex'] = line_line_intersection(data_frame['vector_kaon'], data_frame['point_kaon'],
                                                          data_frame['vector_mu1'], data_frame['point_mu1'])
    data_frame['ktauMu_endvertex'] = line_line_intersection(data_frame['vector_kaon'], data_frame['point_kaon'],
                                                            data_frame['vector_tauMu'], data_frame['point_tauMu'])
    data_frame['mu1tauMu_endvertex'] = line_line_intersection(data_frame['vector_mu1'], data_frame['point_mu1'],
                                                              data_frame['vector_tauMu'], data_frame['point_tauMu'])

    data_frame['pk_PX'] = data_frame['proton_PX'] + data_frame['Kminus_PX']
    data_frame['pk_PY'] = data_frame['proton_PY'] + data_frame['Kminus_PY']
    data_frame['pk_PZ'] = data_frame['proton_PZ'] + data_frame['Kminus_PZ']
    data_frame['pk_vector'] = data_frame[['pk_PX', 'pk_PY', 'pk_PZ']].values.tolist()
    data_frame['pmu1_PX'] = data_frame['proton_PX'] + data_frame['mu1_PX']
    data_frame['pmu1_PY'] = data_frame['proton_PY'] + data_frame['mu1_PY']
    data_frame['pmu1_PZ'] = data_frame['proton_PZ'] + data_frame['mu1_PZ']
    data_frame['pmu1_vector'] = data_frame[['pmu1_PX', 'pmu1_PY', 'pmu1_PZ']].values.tolist()
    data_frame['ptauMu_PX'] = data_frame['proton_PX'] + data_frame['tauMu_PX']
    data_frame['ptauMu_PY'] = data_frame['proton_PY'] + data_frame['tauMu_PY']
    data_frame['ptauMu_PZ'] = data_frame['proton_PZ'] + data_frame['tauMu_PZ']
    data_frame['ptauMu_vector'] = data_frame[['ptauMu_PX', 'ptauMu_PY', 'ptauMu_PZ']].values.tolist()
    data_frame['kmu1_PX'] = data_frame['Kminus_PX'] + data_frame['mu1_PX']
    data_frame['kmu1_PY'] = data_frame['Kminus_PY'] + data_frame['mu1_PY']
    data_frame['kmu1_PZ'] = data_frame['Kminus_PZ'] + data_frame['mu1_PZ']
    data_frame['kmu1_vector'] = data_frame[['kmu1_PX', 'kmu1_PY', 'kmu1_PZ']].values.tolist()
    data_frame['ktauMu_PX'] = data_frame['Kminus_PX'] + data_frame['tauMu_PX']
    data_frame['ktauMu_PY'] = data_frame['Kminus_PY'] + data_frame['tauMu_PY']
    data_frame['ktauMu_PZ'] = data_frame['Kminus_PZ'] + data_frame['tauMu_PZ']
    data_frame['ktauMu_vector'] = data_frame[['ktauMu_PX', 'ktauMu_PY', 'ktauMu_PZ']].values.tolist()
    data_frame['mu1tauMu_PX'] = data_frame['mu1_PX'] + data_frame['tauMu_PX']
    data_frame['mu1tauMu_PY'] = data_frame['mu1_PY'] + data_frame['tauMu_PY']
    data_frame['mu1tauMu_PZ'] = data_frame['mu1_PZ'] + data_frame['tauMu_PZ']
    data_frame['mu1tauMu_vector'] = data_frame[['mu1tauMu_PX', 'mu1tauMu_PY', 'mu1tauMu_PZ']].values.tolist()

    data_frame['pk_ipmu1'] = line_point_distance(vector=data_frame['vector_mu1'], vector_point=data_frame['point_mu1'],
                                                 point=data_frame['pk_endvertex'], direction=data_frame['pk_vector'])
    data_frame['pk_iptauMu'] = line_point_distance(vector=data_frame['vector_tauMu'],
                                                   vector_point=data_frame['point_tauMu'],
                                                   point=data_frame['pk_endvertex'], direction=data_frame['pk_vector'])
    data_frame['pmu1_ipk'] = line_point_distance(vector=data_frame['vector_kaon'],
                                                 vector_point=data_frame['point_kaon'],
                                                 point=data_frame['pmu1_endvertex'],
                                                 direction=data_frame['pmu1_vector'])
    data_frame['pmu1_iptauMu'] = line_point_distance(vector=data_frame['vector_tauMu'],
                                                     vector_point=data_frame['point_tauMu'],
                                                     point=data_frame['pmu1_endvertex'],
                                                     direction=data_frame['pmu1_vector'])
    data_frame['ptauMu_ipk'] = line_point_distance(vector=data_frame['vector_kaon'],
                                                   vector_point=data_frame['point_kaon'],
                                                   point=data_frame['ptauMu_endvertex'],
                                                   direction=data_frame['ptauMu_vector'])
    data_frame['ptauMu_ipmu1'] = line_point_distance(vector=data_frame['vector_mu1'],
                                                     vector_point=data_frame['point_mu1'],
                                                     point=data_frame['ptauMu_endvertex'],
                                                     direction=data_frame['ptauMu_vector'])
    data_frame['kmu1_ipp'] = line_point_distance(vector=data_frame['vector_proton'],
                                                 vector_point=data_frame['point_proton'],
                                                 point=data_frame['kmu1_endvertex'],
                                                 direction=data_frame['kmu1_vector'])
    data_frame['kmu1_iptauMu'] = line_point_distance(vector=data_frame['vector_tauMu'],
                                                     vector_point=data_frame['point_tauMu'],
                                                     point=data_frame['kmu1_endvertex'],
                                                     direction=data_frame['kmu1_vector'])
    data_frame['ktauMu_ipp'] = line_point_distance(vector=data_frame['vector_proton'],
                                                   vector_point=data_frame['point_proton'],
                                                   point=data_frame['ktauMu_endvertex'],
                                                   direction=data_frame['ktauMu_vector'])
    data_frame['ktauMu_ipmu1'] = line_point_distance(vector=data_frame['vector_mu1'],
                                                     vector_point=data_frame['point_mu1'],
                                                     point=data_frame['ktauMu_endvertex'],
                                                     direction=data_frame['ktauMu_vector'])
    data_frame['mu1tauMu_ipp'] = line_point_distance(vector=data_frame['vector_proton'],
                                                     vector_point=data_frame['point_proton'],
                                                     point=data_frame['mu1tauMu_endvertex'],
                                                     direction=data_frame['mu1tauMu_vector'])
    data_frame['mu1tauMu_ipk'] = line_point_distance(vector=data_frame['vector_kaon'],
                                                     vector_point=data_frame['point_kaon'],
                                                     point=data_frame['mu1tauMu_endvertex'],
                                                     direction=data_frame['mu1tauMu_vector'])
    new_cols = ['pk_ipmu1', 'pk_iptauMu', 'pmu1_ipk', 'pmu1_iptauMu', 'ptauMu_ipk', 'ptauMu_ipmu1', 'kmu1_ipp',
                'kmu1_iptauMu', 'ktauMu_ipp', 'ktauMu_ipmu1', 'mu1tauMu_ipp', 'mu1tauMu_ipk']
    return data_frame, new_cols
