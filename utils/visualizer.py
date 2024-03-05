import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import os
from matplotlib import rcParams

rcParams.update({'font.size': 20})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig = plt.figure(figsize=(12, 4))
        ax0 = fig.add_subplot(141)
        ax0.axis('off')
        ax0.imshow(img)
        ax0.set_title('Image')

        ax1 = fig.add_subplot(142)
        ax1.axis('off')
        ax1.imshow(gt, cmap='gray')
        ax1.set_title('GroundTruth')

        ax2 = fig.add_subplot(143)
        ax2.axis('off')
        ax2.imshow(img, cmap='gray', interpolation='none')
        ax2.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax2.set_title('Prediction')

        ax3 = fig.add_subplot(144)
        ax3.axis('off')
        ax3.imshow(vis_img)
        ax3.set_title('Segmentation')

        fig.subplots_adjust
        fig.subplots_adjust(left=0.01,
                            right=0.99,
                            top=0.95,
                            bottom=0.05,
                            wspace=0,
                            hspace=0)  # 调整图像间距
        fig.savefig(os.path.join(save_dir, class_name + '_{}.png'.format(i)),
                    format='png')
        # fig.savefig(os.path.join(save_dir, class_name + '_{}.pdf'.format(i)), format='pdf')
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x
