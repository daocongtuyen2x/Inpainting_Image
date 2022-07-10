from tkinter import *
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from skimage import data
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint

from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace
import skimage
from skimage.measure import label

def draw_mask(image_path, save_mask_path = ""): 
    image_ = Image.open(image_path)
    coordinates = []
    WIDTH, HEIGHT = image_.size
    thickness = int((WIDTH + HEIGHT)*0.0067) + 2
    # print(thickness)

    app = Tk()
    app.geometry(f"{WIDTH}x{HEIGHT}")

    def save(): 
            # filename = filedialog.asksaveasfilename(initialfile="untitle.png", defaultextension="png", filetypes=[("PNG", ".png"), ("JPG", ".jpg")])
            if save_mask_path == "":
                filename_mask = filedialog.asksaveasfilename(initialfile="untitle_mask.png", defaultextension="png", filetypes=[("PNG", ".png"), ("JPG", ".jpg")])
            # print("filename: ", filename)
                if filename_mask != "": 
                    # image_.save(filename_mask)
                    image_mask.save(filename_mask)
            else: 
                image_mask.save(save_mask_path)

    def on_closing(): 
        answer  = messagebox.askyesnocancel("Quit", "Do you want to save your work?", parent = app)
        if answer is not None: 
            if answer: 
                save()
            app.destroy()
            # exit(0)

    def get_x_and_y(event): 
        global lasx, lasy 
        lasx, lasy = event.x, event.y

    def draw_smth(event): 
        global lasx, lasy
        canvas.create_line((lasx, lasy, event.x, event.y), fill = 'black', width = thickness)
        # draw.rectangle([lasx, lasy, event.x, event.y], fill = 'black', width = thickness)
        draw_mask.line([lasx, lasy, event.x, event.y], fill = 'white', width = thickness)
        lasx, lasy = event.x, event.y
        coordinates.append((lasx, lasy))
        # print(lasx, lasy) we can save this to file and use it to draw mask

    canvas = Canvas(app, bg='black')
    canvas.pack(anchor = 'nw', fill = 'both', expand=1)

    canvas.bind("<Button-1>", get_x_and_y)
    canvas.bind("<B1-Motion>", draw_smth)
    image_mask = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    image = ImageTk.PhotoImage(image_)
    # draw = ImageDraw.Draw(image_)
    draw_mask = ImageDraw.Draw(image_mask)
    canvas.create_image(0, 0, image=image, anchor = 'nw')
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.attributes("-topmost", True)
    app.mainloop()

def _get_neighborhood(nd_idx, radius, nd_shape):
    bounds_lo = (nd_idx - radius).clip(min=0)
    bounds_hi = (nd_idx + radius + 1).clip(max=nd_shape)
    return bounds_lo, bounds_hi
 
 
def _inpaint_biharmonic_single_channel(img, mask, out, limits):
    # Initialize sparse matrices
    matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size))
    matrix_known = sparse.lil_matrix((np.sum(mask), out.size))
 
    # Find indexes of masked points in flatten array
    mask_i = np.ravel_multi_index(np.where(mask), mask.shape)
 
    # Find masked points and prepare them to be easily enumerate over
    mask_pts = np.array(np.where(mask)).T
 
    # Iterate over masked points
    for mask_pt_n, mask_pt_idx in enumerate(mask_pts):
        # Get bounded neighborhood of selected radius
        b_lo, b_hi = _get_neighborhood(mask_pt_idx, 2, out.shape)
 
        # Create biharmonic coefficients ndarray
        neigh_coef = np.zeros(b_hi - b_lo)
        neigh_coef[tuple(mask_pt_idx - b_lo)] = 1
        neigh_coef = laplace(laplace(neigh_coef))
 
        # Iterate over masked point's neighborhood
        it_inner = np.nditer(neigh_coef, flags=['multi_index'])
        for coef in it_inner:
            if coef == 0:
                continue
            tmp_pt_idx = np.add(b_lo, it_inner.multi_index)
            tmp_pt_i = np.ravel_multi_index(tmp_pt_idx, mask.shape)
 
            if mask[tuple(tmp_pt_idx)]:
                matrix_unknown[mask_pt_n, tmp_pt_i] = coef
            else:
                matrix_known[mask_pt_n, tmp_pt_i] = coef
 
    # Prepare diagonal matrix
    flat_diag_image = sparse.dia_matrix((out.flatten(), np.array([0])),
                                        shape=(out.size, out.size))
 
    # Calculate right hand side as a sum of known matrix's columns
    matrix_known = matrix_known.tocsr()
    rhs = -(matrix_known * flat_diag_image).sum(axis=1)
 
    # Solve linear system for masked points
    matrix_unknown = matrix_unknown[:, mask_i]
    matrix_unknown = sparse.csr_matrix(matrix_unknown)
    result = spsolve(matrix_unknown, rhs)
 
    # Handle enormous values
    result = np.clip(result, *limits)
 
    result = result.ravel()
 
    # Subssatute masked points with inpainted versions
    for mask_pt_n, mask_pt_idx in enumerate(mask_pts):
        out[tuple(mask_pt_idx)] = result[mask_pt_n]
 
    return out
 
 
def inpaint_biharmonic(img, mask, multichannel=False):
    """Inpaint masked points in image with biharmonic equations.
 
    Parameters
    ----------
    img : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    mask : (M[, N[, ..., P]]) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'img' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    multichannel : boolean, optional
        If True, the last `img` dimension is considered as a color channel,
        otherwise as spatial.
 
    Returns
    -------
    out : (M[, N[, ..., P]][, C]) ndarray
        Input image with masked pixels inpainted.
 
    References
    ----------
    .. [1]  N.S.Hoang, S.B.Damelin, "On surface completion and image inpainting
            by biharmonic functions: numerical aspects",
            http://www.ima.umn.edu/~damelin/biharmonic
 
    Examples
    --------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)
    """
 
    if img.ndim < 1:
        raise ValueError('Input array has to be at least 1D')
 
    img_baseshape = img.shape[:-1] if multichannel else img.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')
 
    if np.ma.isMaskedArray(img):
        raise TypeError('Masked arrays are not supported')
 
    img = skimage.img_as_float(img)
    mask = mask.astype(np.bool)
 
    # Split inpainting mask into independent regions
    kernel = ndi.morphology.generate_binary_structure(mask.ndim, 1)
    mask_dilated = ndi.morphology.binary_dilation(mask, structure=kernel)
    mask_labeled, num_labels = label(mask_dilated, return_num=True)
    mask_labeled *= mask
 
    if not multichannel:
        img = img[..., np.newaxis]
 
    out = np.copy(img)
 
    for idx_channel in range(img.shape[-1]):
        known_points = img[..., idx_channel][~mask]
        limits = (np.min(known_points), np.max(known_points))
 
        for idx_region in range(1, num_labels+1):
            mask_region = mask_labeled == idx_region
            _inpaint_biharmonic_single_channel(
                img[..., idx_channel], mask_region,
                out[..., idx_channel], limits)
 
    if not multichannel:
        out = out[..., 0]
 
    return out

if __name__ == "__main__": 
    path_defected_image = r"tkinker/test6.jpg"
    path_save_mask = r"tkinker/mask_test6.png" #it can be empty. If it empty, tkinter will ask you where to save the generated mask
    # but you need to remember the location to load this mask again in order to compute inpainting algorithm
    path_save_result = r"tkinker/result6.jpg"
    draw_mask(path_defected_image, path_save_mask)

    image_defect = np.asarray(Image.open(path_defected_image))
    image_mask = np.asarray(Image.open(path_save_mask))[:, :, 0]

    #visualize 
    # plt.figure(figsize=(15,15))
    # # plt.subplot(1,3,1), plt.imshow(tulip)
    # plt.subplot(1,3,2), plt.imshow(image_defect)
    # plt.subplot(1,3,3), plt.imshow(image_mask, cmap='gray')

    result_image = inpaint_biharmonic(image_defect, image_mask, multichannel=True)
    imageio.imwrite(path_save_result, result_image)

    #visualize result 
    # plt.rcParams['figure.figsize'] = (30,30)
    # plt.subplot(121), plt.imshow(image_defect)
    # plt.subplot(122), plt.imshow(result_image)

