from background_reduction.b_MC_reduction import b_cleaning
from background_reduction.data_reduction import reduce_background
from data_loader import load_data, add_branches
from get_vertex import get_dimuon_mass, retrieve_vertices, transverse_momentum, line_plane_intersection, \
    tau_momentum_mass, plot_result, plot_b_result

a = load_data(add_branches())
a.dropna(inplace=True)
df = reduce_background(a, True)
# df = a
# df = b_cleaning(a)
df = df.reset_index(drop=True)
df = get_dimuon_mass(df)
df = retrieve_vertices(df)
df = transverse_momentum(df)
df = line_plane_intersection(df)
df = tau_momentum_mass(df)
plot_result(df)
# plot_b_result(df)
