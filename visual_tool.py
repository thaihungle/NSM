from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap



import matplotlib.patches as patches

def plot_meta(rks, wks, rs, ws, data, inlen, error):

    num = len(rks)

    rks = np.squeeze(np.asarray(rks)[:, 0, :],1)
    wks = np.squeeze(np.asarray(wks)[:, 0, :],1)



    rs = np.transpose(np.asarray(rs)[:, 0, :],[1,0])
    ws = np.transpose(np.asarray(ws)[:, 0, :],[1,0])
    N = num

    ax = plt.subplot(5, 1, 1)
    plt.title("Cost per sequence (bits): {}".format(error))

    ax.imshow(np.transpose(np.asarray(data),[1,0]), cmap='hot', interpolation='nearest', aspect='auto')
    if inlen > 0:
        rect = patches.Rectangle( (inlen, 0),
            0,
            data.shape[1], linewidth=0.5, edgecolor='g')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.xlim(-0.5, N - 0.5)
    plt.ylim(0, data.shape[1]-1)

    ax = plt.subplot(5, 1, 2)

    ax.imshow(rs, cmap='hot', interpolation='nearest', aspect='auto')
    plt.ylabel('RDW')
    plt.xlim(-0.5, N - 0.5)

    if inlen>0:
        rect = patches.Rectangle((inlen, 0),
                                 0,
                                 rs.shape[0], linewidth=0.5, edgecolor='g')

        # Add the patch to the Axes
        ax.add_patch(rect)

    ind = np.arange(N)  # the x locations for the groups
    width = 1  # the width of the bars: can also be len(x) sequence
    bot = np.zeros(N)


    plt.subplot(5, 1, 3)
    wnames = []
    for i in range(rks.shape[1]):
        plt.bar(ind, rks[:, i], width,
                bottom=bot)
        bot += rks[:, i]
        wnames.append("program {}".format(i))
    plt.ylabel('RPW')
    plt.ylim(0, 1)
    plt.xlabel('Steps')
    plt.xlim(-0.5, N - 0.5)
    # plt.legend(wnames, fancybox=True, loc='lower center')

    ax = plt.subplot(5, 1, 4)

    ax.imshow(ws, cmap='hot', interpolation='nearest', aspect='auto')

    if inlen>0:
        rect = patches.Rectangle((inlen, 0),
                                 0,
                                 ws.shape[0], linewidth=0.5, edgecolor='g')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.ylabel('WDW')
    plt.xlim(-0.5, N - 0.5)

    plt.subplot(5, 1, 5)
    bot = np.zeros(N)
    for i in range(wks.shape[1]):
        plt.bar(ind, wks[:, i], width,
                bottom=bot)
        bot += wks[:, i]
    plt.ylabel('WPW')
    plt.ylim(0, 1)
    plt.xlabel('Steps')
    plt.xlim(-0.5, N - 0.5)
    # plt.legend(wnames, fancybox=True, loc='lower center')

    plt.show()

def plot_inout(rs, ws, data, inlen, error):

    num = len(data)

    rs = np.transpose(np.asarray(rs)[:, 0, :], [1, 0])
    ws = np.transpose(np.asarray(ws)[:, 0, :], [1, 0])
    N = num

    ax = plt.subplot(3, 1, 1)
    plt.title("Cost per sequence (bits): {}".format(error))

    ax.imshow(np.transpose(np.asarray(data),[1,0]), cmap='hot', interpolation='nearest', aspect='auto')
    if inlen > 0:
        rect = patches.Rectangle( (inlen, 0),
            0,
            data.shape[1], linewidth=0.5, edgecolor='g')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.xlim(-0.5, N - 0.5)
    plt.ylim(0, data.shape[1]-1)

    ax = plt.subplot(3, 1, 2)

    ax.imshow(rs, cmap='hot', interpolation='nearest', aspect='auto')
    plt.ylabel('RDW')
    plt.xlim(-0.5, N - 0.5)

    if inlen>0:
        rect = patches.Rectangle((inlen, 0),
                                 0,
                                 rs.shape[0], linewidth=0.5, edgecolor='g')

        # Add the patch to the Axes
        ax.add_patch(rect)



    ax = plt.subplot(3, 1, 3)

    ax.imshow(ws, cmap='hot', interpolation='nearest', aspect='auto')

    if inlen>0:
        rect = patches.Rectangle((inlen, 0),
                                 0,
                                 ws.shape[0], linewidth=0.5, edgecolor='g')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.ylabel('WDW')
    plt.xlim(-0.5, N - 0.5)


    plt.show()


def plot_state_space(states):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=500, random_state=23)
    new_value1s = tsne_model.fit_transform(states)
    x1 = []
    y1 = []
    for value in new_value1s:
        x1.append(value[0])
        y1.append(value[1])
    alphas = np.linspace(0.1, 1, len(x1))
    rgba_colors = np.zeros((len(x1), 4))
    # for red the first column needs to be one
    rgba_colors[:, 2] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas
    plt.scatter(x1, y1, color = rgba_colors)
    plt.show()