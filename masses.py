import numpy as np
import pandas as pd
from typing import List, Optional

masses = {'mu': 105.658, 'tau': 1777, 'proton': 938.272, 'K': 493.677, 'pi': 139.57, 'D0': 1865,
          'J/psi': 3097, 'psi(2S)': 3686, 'rho0': 770, 'rho1450': 1450, 'kstar': 892,
          'Lc': 2286, 'Lb': 5620, 'B': 5279}


def get_mass(data_frame: pd.DataFrame, particles_associations: List[List[str]], ms:Optional[List]=None) -> pd.DataFrame:
    """
    Obtains the mass of different associations of particles
    :param data_frame:
    :param particles_associations: list of lists made of ['particle_P', 'particle']
    :return:
    """
    if ms is not None:
        for a in range(len(ms)):
            masses[ms[a][0]] = ms[a][1]
    particles = [i[0] for i in particles_associations]
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    return sum_m
