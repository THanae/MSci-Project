import matplotlib.pyplot as plt
import numpy as np

from masses import get_mass, masses


def pp_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'proton']]
    data_frame['pp_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pp_mass'], bins=100, range=[1800, 3000])
        plt.xlabel('$m_{pp}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def kk_mass(data_frame, to_plot):
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K']]
    data_frame['kk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kk_mass'], bins=100, range=[1000, 2500])
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.show()
    data_frame = data_frame[data_frame['kk_mass'] > 1220]
    return data_frame


def plot_pikmu_mass(data_frame, to_plot):
    """
    Plots the pikmu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu']]
    data_frame['pikmu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['pik_mass'] = get_mass(data_frame=data_frame,
                                      particles_associations=[['Kminus_P', 'K'], ['proton_P', 'pi']])
    data_frame['kmu_mass'] = get_mass(data_frame=data_frame,
                                      particles_associations=[['Kminus_P', 'K'], ['mu1_P', 'mu']])
    data_frame['pimu_mass'] = get_mass(data_frame=data_frame,
                                       particles_associations=[['proton_P', 'pi'], ['mu1_P', 'mu']])
    if to_plot:
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pikmu_mass'],
                 bins=75, range=[1500, 5000])
        plt.xlabel('$m_{K\\pi\\mu}$ where pi and mu have the same charge')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pik_mass'],
                 bins=75, range=[500, 4000])
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['pimu_mass'],
                 bins=75, range=[400, 3500])
        plt.xlabel('$m_{\\pi\\mu}$ where p and mu have the same charge')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]['kmu_mass'],
                 bins=75, range=[500, 4000])
        plt.xlabel('$m_{K\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        df_to_plot = data_frame[np.sign(data_frame['mu1_ID']) != np.sign(data_frame['proton_ID'])]
        plt.hist2d(df_to_plot['pikmu_mass'], df_to_plot['kmu_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\mu}$')
        plt.show()
        plt.hist2d(df_to_plot['pikmu_mass'], df_to_plot['pik_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\pi}$')
        plt.show()
        plt.hist(df_to_plot['kmu_mass'], range=[[1500, 4500], [1000, 3000]], bins=40)
        plt.xlabel('$m_{\\pi K\\mu}$')
        plt.ylabel('$m_{K\\mu}$')
        plt.show()
    return data_frame


def plot_pikmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pikmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'pi']]
    data_frame['pik_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        # data_frame = data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]
        plt.hist(data_frame['pikmumu_mass'], bins=100, range=[3500, 6500])
        plt.axvline(masses['B'], c='k')
        plt.axvline(5366, c='k')  # Bs mass
        plt.xlabel('$m_{K\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        test_part = data_frame[
            (data_frame['pikmumu_mass'] < masses['B'] + 100) & (data_frame['pikmumu_mass'] > masses['B'] - 100)]
        plt.hist(data_frame['pik_mass'], bins=50, range=[500, 2000])
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(test_part['pik_mass'], bins=100, range=[500, 2000])
        plt.axvline(masses['kstar'], c='k')
        plt.xlabel('$m_{K\\pi}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist2d(data_frame['pikmumu_mass'], data_frame['pik_mass'], bins=30, range=[[2000, 7000], [500, 2000]])
        plt.xlabel('$m_{K\\pi\\mu\\mu}$')
        plt.ylabel('$m_{K\\pi}$')
        # plt.savefig('pikmumu_pik_nolb.png')
        plt.show()
    to_remove_b = data_frame[
        ((data_frame['pikmumu_mass'] < 5366 + 200) & (data_frame['pikmumu_mass'] > masses['B'] - 200)) & (
                (data_frame['pik_mass'] > masses['kstar'] - 30) & (data_frame['pik_mass'] < masses['kstar'] + 30))]
    data_frame = data_frame.drop(to_remove_b.index)
    return data_frame


def plot_kmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['p(k)mumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['mu1_P', 'mu'], ['proton_P', 'mu']]
    data_frame['kmup(mu)_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kmumu_mass'], bins=100)
        plt.xlabel('$m_{k\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['p(k)mumu_mass'], bins=100)
        plt.xlabel('$m_{p(k)\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['kmup(mu)_mass'], bins=100)
        plt.xlabel('$m_{k\\mu p(\\mu )}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_kpmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kpmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'proton'], ['proton_P', 'K']]
    data_frame['kp_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kpmumu_mass'], bins=100, range=[3500, 8000])
        plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['kpmumu_mass'],
                 bins=100, range=[3500, 8000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{kp\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame[(data_frame['Lb_M'] < 5620 - 40) | (data_frame['Lb_M'] > 5620 + 40)]['kpmumu_mass'],
                 bins=100, range=[3500, 8000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{kp\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist2d(data_frame['kpmumu_mass'], data_frame['kp_mass'], bins=50, range=[[3500, 6500], [1450, 3000]])
        plt.show()
    data_frame = data_frame[(data_frame['kpmumu_mass'] < 5620 - 40) | (data_frame['kpmumu_mass'] > 5620 + 40)]
    return data_frame


def plot_pipimumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'pi'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pipimumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'pi']]
    data_frame['pipi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    data_frame['mumu_mass'] = get_mass(data_frame=data_frame,
                                       particles_associations=[['mu1_P', 'mu'], ['tauMu_P', 'mu']])
    if to_plot:
        plt.hist2d(data_frame['pipimumu_mass'], data_frame['pipi_mass'], bins=30, range=[[2000, 7000], [200, 2000]])
        plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
        plt.ylabel('$m_{\\pi\\pi}$')
        plt.show()
        plt.show()
        plt.hist(data_frame['pipimumu_mass'], bins=100, range=[2000, 7000])
        plt.xlabel('$m_{\\pi\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['B'], c='k')
        plt.show()
        plt.hist(data_frame['pipi_mass'], bins=100, range=[200, 2000])
        plt.xlabel('$m_{\\pi\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['K'], c='k')
        plt.show()
        plt.hist(data_frame['mumu_mass'], bins=100, range=[200, 2000])
        plt.xlabel('$m_{\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(masses['K'], c='k')
        plt.show()
    data_frame = data_frame[
        (data_frame['pipimumu_mass'] < masses['B']) | (data_frame['pipimumu_mass'] > masses['B'] + 100)]
    return data_frame


def plot_ppimumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['ppimumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'pi'], ['proton_P', 'proton']]
    data_frame['ppi_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        test_data = data_frame[(data_frame['ppimumu_mass'] > 5400) & (data_frame['ppimumu_mass'] < 5600)]
        plt.hist(data_frame['ppimumu_mass'], bins=150, range=[4000, 7000])
        plt.hist(data_frame[(data_frame['Lb_M'] > 5620 - 40) & (data_frame['Lb_M'] < 5620 + 40)]['ppimumu_mass'],
                 bins=150, range=[3000, 7000])
        plt.axvline(masses['Lb'], c='k')
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
        plt.hist(data_frame['ppi_mass'], bins=100, range=[1000, 3000])
        plt.xlabel('$m_{p\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(1115, c='k')
        plt.show()
        plt.hist(test_data['ppi_mass'], bins=100, range=[1000, 3000])
        plt.xlabel('$m_{p\\pi}$')
        plt.ylabel('occurrences')
        plt.axvline(1115, c='k')
        plt.show()
        plt.hist2d(data_frame['ppimumu_mass'], data_frame['ppi_mass'], bins=30, range=[[3000, 7000], [1000, 4000]])
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('$m_{p\\pi}$')
        plt.axvline(masses['Lb'], c='k')
        plt.axhline(1115, c='k')
        plt.show()
        plt.hist2d(data_frame['ppimumu_mass'], data_frame['Lb_M'], bins=60, range=[[3000, 7000], [3000, 7000]])
        plt.xlabel('$m_{p\\pi\\mu\\mu}$')
        plt.ylabel('$m_{Lb}$')
        plt.show()
    data_frame = data_frame[(data_frame['ppimumu_mass'] < 5420) | (data_frame['ppimumu_mass'] > 5620)]
    return data_frame


def plot_kkmumu_mass(data_frame, to_plot):
    """
    Plots the pikmumu mass where the proton has a pion hypothesis
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['kkmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'K']]
    data_frame['kk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['kkmumu_mass'], bins=100, range=[2500, 7000])
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(5366, c='k')  # Bs mass
        plt.show()
        test_data = data_frame[(data_frame['kkmumu_mass'] > 5366 - 100) & (data_frame['kkmumu_mass'] < 5366 + 100)]
        plt.hist(data_frame['kk_mass'], bins=100, range=[900, 2000])
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.axvline(1020, c='k')
        plt.show()
        plt.hist(test_data['kk_mass'], bins=100, range=[900, 2000])
        plt.xlim(right=2000)
        plt.xlabel('$m_{KK}$')
        plt.ylabel('occurrences')
        plt.axvline(1020, c='k')
        plt.show()
        plt.hist(test_data['kkmumu_mass'], bins=100)
        plt.xlim(right=7000)
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.axvline(5366, c='k')
        plt.show()
        plt.hist2d(data_frame['kkmumu_mass'], data_frame['kk_mass'], bins=30, range=[[2000, 7000], [900, 2000]])
        plt.axvline(5366, c='k')
        plt.axhline(1020, c='k')
        plt.xlabel('$m_{KK\\mu\\mu}$')
        plt.ylabel('$m_{KK}$')
        plt.show()
    to_drop = data_frame[((data_frame['kkmumu_mass'] > 5366 - 200) & (data_frame['kkmumu_mass'] < 5366 + 200)) & (
            (data_frame['kk_mass'] > 990) & (data_frame['kk_mass'] < 1050))]
    data_frame = data_frame.drop(to_drop.index)
    return data_frame


def plot_pmumu_mass(data_frame, to_plot=True):
    """
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['proton_P', 'proton'], ['mu1_P', 'mu'], ['tauMu_P', 'mu']]
    data_frame['pmumu_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        plt.hist(data_frame['pmumu_mass'], bins=100)
        plt.xlabel('$m_{p\\mu\\mu}$')
        plt.ylabel('occurrences')
        plt.show()
    return data_frame


def plot_pmu_mass(data_frame):
    """
    Plots the pmu mass
    :param data_frame:
    :return:
    """
    particles_associations = [['proton_P', 'proton'], ['mu1_P', 'mu']]
    data_frame['pmu'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    df_to_plot = data_frame[np.sign(data_frame['proton_ID']) == np.sign(data_frame['mu1_ID'])]
    plt.hist(df_to_plot['pmu'], bins=100, range=[1400, 3700])
    plt.xlim(right=3700)
    plt.xlabel('$m_{p\\mu}$')
    plt.ylabel('occurrences')
    plt.show()
    return data_frame


def plot_pk_mass(data_frame, to_plot=True):
    """
    Plots the pk mass
    :param data_frame:
    :param to_plot:
    :return:
    """
    particles_associations = [['Kminus_P', 'K'], ['proton_P', 'proton']]
    data_frame['pk_mass'] = get_mass(data_frame=data_frame, particles_associations=particles_associations)
    if to_plot:
        df_with_signs = data_frame[np.sign(data_frame['Kminus_ID']) != np.sign(data_frame['proton_ID'])]
        plt.hist(df_with_signs['pk_mass'], bins=120, range=[1400, 2600])  # peak 1480 to 1555
        plt.xlabel('$m_{pK}$')
        plt.ylabel('occurrences')
        plt.show()
        a = data_frame[(data_frame['pk_mass'] > 1480) & (data_frame['pk_mass'] < 1550)]
        plt.hist(a['pKmu_ENDVERTEX_CHI2'], bins=50)
        plt.xlabel('pKmu_ENDVERTEX_CHI2')
        plt.show()

    # data_frame = data_frame[(data_frame['pk_mass'] < 1480) | (data_frame['pk_mass'] > 1550)]
    # data_frame = data_frame[(data_frame['pk_mass'] < 1500) | (data_frame['pk_mass'] > 1540)]
    # data_frame = data_frame[(data_frame['pk_mass'] > 1910) | (data_frame['pk_mass'] < 1850)]
    return data_frame


def jpsi_swaps(data_frame):
    """
    Look for jpsi swaps
    :param data_frame:
    :return:
    """
    particles_duos = [['mu1_P', 'tauMu_P'], ['mu1_P', 'proton_P'], ['mu1_P', 'Kminus_P'], ['tauMu_P', 'proton_P'],
                      ['tauMu_P', 'Kminus_P'], ['proton_P', 'Kminus_P']]
    for p1, p2 in particles_duos:
        particles_associations = [[p1, 'mu'], [p2, 'mu']]
        if ('mu' in p1.lower() and 'mu' in p2.lower()) or ('mu' not in p1.lower() and 'mu' not in p2.lower()):
            mass_frame = data_frame[np.sign(data_frame[p1[:-2] + '_ID']) != np.sign(data_frame[p2[:-2] + '_ID'])]
        else:
            mass_frame = data_frame[np.sign(data_frame[p1[:-2] + '_ID']) == np.sign(data_frame[p2[:-2] + '_ID'])]
        mass = get_mass(data_frame=mass_frame, particles_associations=particles_associations)
        plt.hist(mass, bins=100)
        plt.axvline(masses['J/psi'], c='k')
        plt.xlabel(p1[:-2] + ' ' + p2[:-2])
        plt.xlim(right=4500)
        plt.show()
    return data_frame
