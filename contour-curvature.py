import cv2
import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as pl
from tqdm import tqdm


# size of pixels for comparable gradient computations between resolutions
pixel_size = 0.5

print("create random blobs")
shape = 10 * np.array((108, 192))
noise = np.random.normal(size=shape)
noise = ndi.gaussian_filter(noise, sigma=10)
binar = (noise > np.percentile(noise.ravel(), 75)).astype("int")
binar = ndi.binary_dilation(binar, iterations=1)

print("label blobs")
label, _ = ndi.label(binar)
label = label.astype("float")
label[label == 0] = np.nan

print("compute contours and their curvature")
crv_label = np.ones(label.shape) * np.nan
ul = np.unique(label[~np.isnan(label)])

for li in tqdm(ul):
    # use opencv to find contour polygon
    contours, _ = cv2.findContours((label == li).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    contour = contour.reshape(-1, 2)
    x, y = contour[:, 0], contour[:, 1]
    xe = np.tile(x, 3)
    ye = np.tile(y, 3)
    plen = len(x)

    # smooth x,y coords for a more non-local curvature
    xe = ndi.gaussian_filter1d(xe.astype("float"), sigma=3)
    ye = ndi.gaussian_filter1d(ye.astype("float"), sigma=3)

    # https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization
    dx = np.gradient(xe, pixel_size)
    dy = np.gradient(ye, pixel_size)
    ddx = np.gradient(dx, pixel_size)
    ddy = np.gradient(dy, pixel_size)
    dx = dx[plen:-plen]
    dy = dy[plen:-plen]
    ddx = ddx[plen:-plen]
    ddy = ddy[plen:-plen]
    crv = (dx * ddy - dy * ddx) / np.sqrt((dx * dx + dy * dy) ** 3)

    # write into image
    crv_label[y, x] = crv

print("figure")
thr = np.nanpercentile(crv_label.ravel(), 95)
fg, ax = pl.subplots()
im = ax.imshow(
    crv_label, origin="lower", interpolation="none", cmap="PiYG", vmin=-thr, vmax=thr
)
fg.colorbar(im, ax=ax, shrink=0.8).set_label("contour curvature")
pl.tight_layout()
pl.show()
