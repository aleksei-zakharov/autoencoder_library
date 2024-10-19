import matplotlib.pyplot as plt
N1 = 5  # option tenor index (10Y)
N2 = 4  # swap tenor index (10Y)
N3 = 2  # strike index (ATM)

def plot_real_vs_prediction_vols(predictions,
                                 real_vols,
                                 indexes):
    idx1, idx2, idx3 = indexes
    plt.figure(figsize=(15,6))
    plt.plot(predictions[:, idx1, idx2, idx3], label='predictions')
    plt.plot(real_vols[:, idx1, idx2, idx3], label='real vols')
    plt.ylabel('vol in bp')
    plt.xlabel('dates')
    plt.legend()
    plt.show()