import numpy as np

def voc_colormap(N=256):
    def bitget(val, idx): return ((val & (1 << idx)) != 0)
 
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        # print([r, g, b])
        cmap[i, :] = [r, g, b]
    return cmap
 

mvtec_colormap = {
    'bottle': [128, 0, 0],
    'cable': [0, 128, 0],
    'capsule': [128, 128, 0],
    'carpet': [0, 0, 128],
    'grid': [128, 0, 128],
    'hazelnut': [0, 128, 128],
    'leather': [128, 128, 128],
    'metal_nut': [64, 0, 0],
    'pill': [192, 0, 0],
    'screw': [64, 128, 0],
    'tile': [192, 128, 0],
    'toothbrush': [64, 0, 128],
    'transistor': [192, 0, 128],
    'wood': [192, 128, 128],
    'zipper': [0, 64, 128]
}


if __name__=='__main__':
    # VOC_COLORMAP = voc_colormap()
    # print('sss')
    pass
