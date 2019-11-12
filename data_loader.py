import uproot


def load_data(columns, year=2016):
    columns = set(columns)
    file = uproot.open(f"merge_{year}.root")
    events = file['LbTuple/DecayTree']
    events_frame = events.pandas.df(columns)
    print(events.show())
    print(events_frame.index)
    # events_frame = events_frame.reset_index(level=['subentry'])
    # no sub-entries when using ownpv so sub-entries have something to do with other primary vertices
    # events_frame = events_frame[events_frame['subentry'] < 1]
    # events_frame = events_frame.drop('subentry', axis=1)
    return events_frame


def add_branches():
    """
    Returns branches needed for analysis - for now returns branches used by Matt
    :return:
    """
    lb = ["Lb_M", "Lb_ENDVERTEX_CHI2", "Lb_DIRA_OWNPV"]
    primary = ['Lb_OWNPV_X', 'Lb_OWNPV_Y', 'Lb_OWNPV_Z', 'Lb_OWNPV_XERR', 'Lb_OWNPV_YERR', 'Lb_OWNPV_ZERR']
    end_vertex = ['Lb_ENDVERTEX_X', 'Lb_ENDVERTEX_Y', 'Lb_ENDVERTEX_Z', 'Lb_ENDVERTEX_XERR', 'Lb_ENDVERTEX_YERR',
                  'Lb_ENDVERTEX_ZERR']
    final_ipchi2 = ["mu1_IPCHI2_OWNPV", "mu2_IPCHI2_OWNPV", "proton_IPCHI2_OWNPV", "Kminus_IPCHI2_OWNPV"]
    pid = ["Kminus_PIDK", "Kminus_PIDp", "Kminus_PIDd", "proton_PIDK", "proton_PIDp", "proton_PIDd", "Kminus_PIDmu",
           "mu1_PIDK", "mu2_PIDK", "mu1_PIDmu", "mu2_PIDmu"]
    bdt_isolation = ["Lb_pmu_ISOLATION_BDT1"]
    cleaning = ["proton_P", "proton_PT", "Kminus_PT", "mu1_P", "mu1_PT", "mu2_P", "mu2_PT"]
    proton = ["proton_PE", "proton_PX", "proton_PY", "proton_PZ", "proton_P"]
    kminus = ["Kminus_PE", "Kminus_PX", "Kminus_PY", "Kminus_PZ", "Kminus_P"]
    mu1 = ["mu1_PE", "mu1_PX", "mu1_PY", "mu1_PZ", "mu1_P"]
    mu2 = ["mu2_PE", "mu2_PX", "mu2_PY", "mu2_PZ", "mu2_P"]
    return lb + final_ipchi2 + pid + bdt_isolation + cleaning + proton + kminus + mu1 + mu2 + primary + end_vertex + [
        "Lb_PE", "Lb_PX", "Lb_PY", "Lb_PZ", "Lb_P"]


if __name__ == '__main__':
    a = load_data(add_branches())
    print(a)
