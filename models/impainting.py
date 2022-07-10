import numpy as np
import scipy.ndimage as ndi 
from skimage.measure import label, regionprops # measure
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage.filters import laplace
from skimage import img_as_float
import os
from PIL import Image

from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace
import skimage
from skimage.measure import label

PATH_TO_TEST = 'image_test'
PATH_TO_RESULT = 'image_result'

#impaiting('test1.jpg', 30)
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
    
 
    if img.ndim < 1:
        raise ValueError('Input array has to be at least 1D')
 
    img_baseshape = img.shape[:-1] if multichannel else img.shape
    if img_baseshape != mask.shape:
        print(img_baseshape)
        print(mask.shape)
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

def biharmonic(img_path, n_holes=20, check = True):
    file_name = img_path.split('/')[-1]
    img = Image.open(img_path)
    img = np.array(img)
    img_hole = img.copy()
    for i in range(n_holes):
        pos = np.random.randint(20, img.shape[0]-20, 2)
        hole_size = np.random.randint(1, 50, 2)
        img_hole[pos[0]:pos[0]+hole_size[0], pos[1]:pos[1]+hole_size[1]] = (0,0,0) # gán khu vực mask = màu đen.
    hole_mask = np.bitwise_and(np.bitwise_and(img_hole[:,:,0] == 0, img_hole[:,:,1] == 0), img_hole[:,:,2]==0)
    img_result = inpaint_biharmonic(img_hole, hole_mask, check)
    path_result = 're'+file_name
    img_hole_save = Image.fromarray((img_hole.astype(np.uint8)))
    img_hole_save.save('re'+'image_hole.jpg')
    img_result = Image.fromarray((img_result * 255).astype(np.uint8))
    img_result.save(path_result)

def test_biharmonic(img_path, mask_path, check = True):
    file_name = img_path.split('/')[-1]
    img = Image.open(img_path)
    img = np.array(img)
    mask = Image.open(mask_path)
    background = Image.new("RGB", mask.size, (255, 255, 255))
    background.paste(mask, mask = mask.split()[3])
    mask = background
    mask = np.array(mask)
    mask = np.where(mask==0,True, False)
    mask_one = mask[:,:,0]
    zero_matrix = np.zeros_like(img)
    img_hole = np.where(mask,zero_matrix,img)
    print('start impaint!')
    img_result = inpaint_biharmonic(img_hole, mask_one, check)
    path_result = 'static/uploads/result.jpg'
    img_result = Image.fromarray((img_result * 255).astype(np.uint8))
    img_result.save(path_result)



if __name__ == "__main__":
    test_biharmonic('image_test/test6.jpg', 'image_test/mask.jpg')
    # biharmonic('image_test/test6.jpg')
