import matplotlib.pyplot as plt
import numpy as np
import os

def plot_band_powers(alpha_powers, theta_powers, labels):
    # plot alpha/theta powers (36 epochs) on a 2D feature space
    # inputs: alpha/theta_powers -> array of powers for each epoch, labels -> 0 for eyes open (1 for closed)

    # may not need these lines
    alpha_powers = np.array(alpha_powers)
    theta_powers = np.array(theta_powers)
    labels = np.array(labels)

    plt.figure(figsize=(7, 6))
    plt.scatter(alpha_powers[labels == 0], theta_powers[labels == 0], color='blue', label = 'Eyes open', alpha=0.7) # x, y...
    plt.scatter(alpha_powers[labels == 1], theta_powers[labels == 1], color='red', label = 'Eyes closed', alpha=0.7)
    plt.xlabel("Mean alpha band power [dB]")
    plt.ylabel("Mean theta band power [dB]")
    # plt.title("EEG epoch power feature space")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # save plot
    file_name = "plots/epoch_powers.png"
    plt.savefig(file_name, dpi = 300)

    plt.close()
 
    return
