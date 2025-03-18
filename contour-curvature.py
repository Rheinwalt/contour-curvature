import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as pl
from scipy.spatial import cKDTree as kdtree
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

print("find contours")
contour_label = label.copy()
contour_label[ndi.binary_erosion(~np.isnan(label))] = np.nan

print("compute contour curvature")
crv_label = np.ones(label.shape) * np.nan
cy, cx = np.nonzero(~np.isnan(contour_label))
cl = contour_label[cy, cx]

for li in tqdm(np.unique(cl)):
    # vectorize contour li
    i = np.where(cl == li)[0]
    if len(i) < 10:
        # no curvature for small polygons
        continue
    y, x = cy[i], cx[i]
    pt = np.c_[x, y]
    tr = kdtree(pt)
    k = 4
    _, ii = tr.query(pt, k=k, workers=-1)
    ir = [
        ii[0, 1],
    ]
    queue = [
        ii[0, 0],
    ]
    while len(queue):
        i = queue.pop(0)
        for j in range(k):
            if ii[i, j] not in ir:
                queue.append(ii[i, j])
                ir.append(ii[i, j])
                break

    x, y = x[ir], y[ir]

    # fix orientation of contour polygon
    j = np.argmin(y[x == x.min()])
    jr = np.arange(len(x))[x == x.min()]
    j = jr[j]
    xe = np.concat(
        (
            [
                x[-1],
            ],
            x,
            [
                x[0],
            ],
        )
    )
    ye = np.concat(
        (
            [
                y[-1],
            ],
            y,
            [
                y[0],
            ],
        )
    )
    j += 1
    det = ((xe[j] - xe[j - 1]) * (ye[j + 1] - ye[j - 1])) - (
        (xe[j + 1] - xe[j - 1]) * (ye[j] - ye[j - 1])
    )
    if det > 0:
        xe, ye = xe[::-1], ye[::-1]

    # smooth x,y coords for a more non-local curvature
    xe = ndi.gaussian_filter1d(xe.astype("float"), sigma=3)
    ye = ndi.gaussian_filter1d(ye.astype("float"), sigma=3)

    # https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization
    dx = np.gradient(xe, pixel_size)
    dy = np.gradient(ye, pixel_size)
    ddx = np.gradient(dx, pixel_size)
    ddy = np.gradient(dy, pixel_size)
    dx = dx[1:-1]
    dy = dy[1:-1]
    ddx = ddx[1:-1]
    ddy = ddy[1:-1]
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
