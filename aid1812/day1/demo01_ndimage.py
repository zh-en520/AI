"""
demo01_ndimage.py   图像简单处理
"""
import scipy.misc as sm
import scipy.ndimage as sn
import matplotlib.pyplot as mp

img = sm.imread('../da_data/lily.jpg',True)
#高斯模糊
img2 = sn.median_filter(img, 20)
#角度旋转
img3 = sn.rotate(img, 45)
#边缘识别
img4 = sn.prewitt(img)
mp.figure('ndimage', facecolor='lightgray')
mp.subplot(221)
mp.xticks([])
mp.yticks([])
mp.imshow(img, cmap='gray')

mp.subplot(222)
mp.xticks([])
mp.yticks([])
mp.imshow(img2, cmap='gray')

mp.subplot(223)
mp.xticks([])
mp.yticks([])
mp.imshow(img3, cmap='gray')

mp.subplot(224)
mp.xticks([])
mp.yticks([])
mp.imshow(img4, cmap='gray')
mp.tight_layout()
mp.show()






