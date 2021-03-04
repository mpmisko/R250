import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

def plot_stats(agent1_rewards, agent2_rewards, i_episode):
    fig, axs = plt.subplots(2)
    r1 = np.array(agent1_rewards)
    r2 = np.array(agent2_rewards)
    #sns.lineplot(data=r1, ax=axs[0])
    #sns.lineplot(data=r2, ax=axs[1])
    sns.lineplot(data=gaussian_filter1d(r1, 50), ax=axs[0])
    sns.lineplot(data=gaussian_filter1d(r2, 50), ax=axs[1])
    plt.savefig('./training.png')