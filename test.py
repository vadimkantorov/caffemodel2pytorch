import torch
import caffemodel2pytorch

from torchsummary import summary
import skimage

import SimpleITK as sitk

import matplotlib.pyplot as plt

model = caffemodel2pytorch.Net(
    prototxt="../CDSR/x2/CDSRx2.prototxt",
    weights="../CDSR/x2/solver1_iter_104192.caffemodel",
    caffe_proto="http://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto",
)
print(model)

img = sitk.ReadImage("../CDSR/test_data/KKI2009-01-MPRAGE.nii")
img_arr = sitk.GetArrayFromImage(img)

img_arr -= img_arr.min()
img_arr /= img_arr.max()

img_down2 = skimage.transform.rescale(img_arr, scale=0.5, anti_aliasing=True)
img_down2 -= img_down2.min()
img_down2 /= img_down2.max()
img_up2 = skimage.transform.rescale(img_down2, scale=2.0, anti_aliasing=True, order=1)

img_down3 = skimage.transform.rescale(img_arr, scale=0.3333333, anti_aliasing=True)
img_down3 -= img_down3.min()
img_down3 /= img_down3.max()
img_up3 = skimage.transform.rescale(img_down3, scale=3.0, anti_aliasing=True, order=1)

super_res = model(img_up2[None, None, ...])

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].imshow(img_arr[170, ...])
ax[0, 1].imshow(img_up2[170, ...])
ax[1, 0].imshow(img_up3[170, ...])
ax[1, 1].imshow(super_res)
plt.show()
