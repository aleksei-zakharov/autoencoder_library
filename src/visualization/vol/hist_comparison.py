import matplotlib.pyplot as plt
import os


def hist_comparison(data1,
                    data1_label,
                    data2,
                    data2_label,
                    alpha=0.8,
                    bins_all=500,
                    threshold=140,
                    bins_higher_th=20,
                    save_name=None
                    ):
    LEGEND_FONTSIZE = 10

    plt.figure(figsize=(12, 4.5))
    plt.suptitle('Histograms of volatilities')

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(f'Histogram of {data1_label} and {data2_label} volatilities')
    ax1.hist(data1.reshape(-1), bins=bins_all, density=True, label=data1_label, alpha = alpha, color='orange')
    ax1.hist(data2.reshape(-1), bins=bins_all, density=True, label=data2_label, alpha = alpha)
    ax1.set_xlabel('bp')
    ax1.set_ylabel('pdf')
    ax1.legend(fontsize=LEGEND_FONTSIZE)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f'Histogram of volatilities larger than {threshold}bp')
    ax2.hist(data1[data1>threshold].reshape(-1), bins=bins_higher_th, density=True, label=data1_label, alpha=alpha, color='orange')
    ax2.hist(data2[data2>threshold].reshape(-1), bins=bins_higher_th, density=True, label=data2_label, alpha=alpha)
    ax2.set_xlabel('bp')
    ax2.set_ylabel('pdf')
    ax2.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/vol'
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, 'train_test_histograms.png')
        plt.savefig(file_path)

    # Display the plot
    plt.show()