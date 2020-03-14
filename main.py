from background_reduction.b_MC_reduction import b_cleaning
from background_reduction.data_reduction import reduce_background
from data.data_loader import load_data
from get_vertex import obtain_lb_line_of_flight, transverse_momentum, line_plane_intersection, tau_momentum_mass, \
    plot_result, plot_b_result

df_name = 'Lb_data'
# df_name = 'B_MC'

a = load_data(df_name=df_name)
a.dropna(inplace=True)
if df_name == 'Lb_data':
    df = reduce_background(a, True)
elif df_name == 'B_MC':
    df = b_cleaning(a)
else:
    df = a
df = df.reset_index(drop=True)
df = obtain_lb_line_of_flight(df)
df = transverse_momentum(df)
df = line_plane_intersection(df)
df = tau_momentum_mass(df)
if df_name != 'B_MC':
    plot_result(df)
else:
    plot_b_result(df)
