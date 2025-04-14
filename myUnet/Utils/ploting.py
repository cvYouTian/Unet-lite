from matplotlib import pyplot as plt


def input_images(x, y, i, n_iter, k):
    if k == 1:
        x1 = x
        y1 = y

        x2 = x1.to("cpu")
        y2 = y1.to("cpu")
        x2 = x2.detach()
        y2 = y2.detach()

        x3 = x2[1, 1, :, :]
        y3 = y2[1, 0, :, :]

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.imshow(y3)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.savefig(
            ...
        )