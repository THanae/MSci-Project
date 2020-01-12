import numpy as np

masses = {'mu': 105.658,  'tau': 1777, 'proton': 938.272, 'K': 493.677, 'pi': 139.57, 'D0': 1865,
          'J/psi': 3097, 'psi(2S)': 3686, 'rho0': 770, 'rho1450': 1450, 'kstar': 892,
          'Lc': 2286, 'Lb': 5620, 'B': 5279}


def get_mass(data_frame, particles_associations):
    particles = [i[0] for i in particles_associations]
    energy = sum([np.sqrt(data_frame[i] ** 2 + masses[j] ** 2) for i, j in particles_associations])
    mom_x = sum([data_frame[i + 'X'] for i in particles])
    mom_y = sum([data_frame[i + 'Y'] for i in particles])
    mom_z = sum([data_frame[i + 'Z'] for i in particles])
    sum_m = np.sqrt(energy ** 2 - mom_x ** 2 - mom_y ** 2 - mom_z ** 2)
    return sum_m
