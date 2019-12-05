import matplotlib.pyplot as plt


def plot_compare_data(control_data, background_data, histogram_range, columns_to_plot, signal_name = 'signal'):
    fig, axs = plt.subplots(len(columns_to_plot), 2, gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
    bins = 50
    for i in range(len(columns_to_plot)):
        column = columns_to_plot[i]
        if isinstance(column, str):
            axs[i, 0].hist(control_data[column], bins=bins, range=[0, histogram_range])
            axs[i, 0].set_title(column + ' ' + signal_name)
            axs[i, 1].hist(background_data[column], bins=bins, range=[0, histogram_range])
            axs[i, 1].set_title(column + ' background')
        else:
            axs[i, 0].hist(control_data[column[0]] - control_data[column[1]], bins=bins, range=[0, histogram_range])
            axs[i, 0].set_title(column[0] + ' - ' + column[1] + ' ' + signal_name)
            axs[i, 1].hist(background_data[column[0]] - background_data[column[1]], bins=bins, range=[0, histogram_range])
            axs[i, 1].set_title(column[0] + ' - ' + column[1] + ' background')
    plt.show()
