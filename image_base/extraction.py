from joblib import Parallel, delayed

import numpy as np
from . import io_utils


def extract_patches(img, tilesize, steps_per, return_coords=False):
    #ch ro co
    step_size = int(tilesize/steps_per)
    img_shape = img.shape
    if len(img_shape) == 2:
        if io_utils.BACKEND == 'th':
            img = img.reshape((1,) + img_shape)
        else:
            img = img.reshape(img_shape + (1,))

    if io_utils.BACKEND != 'th':
        img = img.transpose(2,0,1)
    rows, cols = img.shape[1:]

    y_coords = list(range(0, rows, step_size))
    x_coords = list(range(0, cols, step_size))
    patches = []
    coords = []
    for ind_y in range(len(y_coords) - steps_per):
        y0 = y_coords[ind_y]
        y1 = y_coords[ind_y + steps_per]
        for ind_x in range(len(x_coords) - steps_per):
            x0 = x_coords[ind_x]
            x1 = x_coords[ind_x + steps_per]
            patches.append(img[:, y0:y1, x0:x1])
            coords.append((y0,x0))

    if rows % tilesize > 0:
        y0 = rows - tilesize
        y1 = rows
        for ind_x in range(len(x_coords) - steps_per):
            x0 = x_coords[ind_x]
            x1 = x_coords[ind_x + steps_per]
            patches.append(img[ :, y0:y1, x0:x1])
            coords.append((y0, x0))

    if cols % tilesize > 0:
        x0 = cols - tilesize
        x1 = cols
        for ind_y in range(len(y_coords) - steps_per):
            y0 = y_coords[ind_y]
            y1 = y_coords[ind_y + steps_per]
            patches.append(img[:, y0:y1, x0:x1])
            coords.append((y0, x0))

    patches = [patch.reshape(1,-1,tilesize,tilesize) for patch in patches]

    if io_utils.BACKEND != 'th':
        patches = [patch.transpose(0, 2, 3, 1) for patch in patches]

    patches = np.concatenate(patches)
    if len(img_shape)==2:
        patches = patches.reshape((-1,tilesize,tilesize))
    if return_coords:
        return patches, coords
    else:
        return patches


def blockshaped(arr, tilesize, flat=False, steps_per=2, n_jobs=3):

    tiles = Parallel(n_jobs=n_jobs)(delayed(extract_patches)(arr[ind],tilesize, steps_per) for ind in range(len(arr)))
    tiles = np.concatenate(tiles, axis=0)
    if flat:
        tiles = np.reshape(tiles, (len(tiles),-1))
    return tiles

def tile_targets(tile, n_dim=4):
    tilesize = int(tile.shape[2]/n_dim)
    extracted = extract_patches(tile, tilesize, steps_per=1)
    targets = extracted.reshape(1, n_dim**2,-1).mean(axis=2)
    return targets


def blockshapedy_repeat(arr,y, tilesize, steps_per=2, n_jobs=3):
    nper = extract_patches(arr[0], tilesize, steps_per).shape[0]

    tiles = Parallel(n_jobs=n_jobs)(delayed(extract_patches)(arr[ind], tilesize, steps_per) for ind in range(len(arr)))
    tiles = np.concatenate(tiles, axis=0)
    y_rep = np.repeat(y, nper, axis=0)
    return tiles, y_rep

def blockshapedy_transform(tiles, y_rep, n_dim=4, zeroind=4, n_jobs=3):

    targets = Parallel(n_jobs=n_jobs)(delayed(tile_targets)(tile, n_dim=n_dim) for tile in tiles)

    yperc = np.concatenate(targets, axis=0)

    #yperc = np.reshape(tiles, (len(tiles),-1)).mean(axis=1).reshape(-1,1)
    yperc2 = np.repeat(yperc.mean(axis=1).reshape((-1,1)), y_rep.shape[1],axis=1)*y_rep>0.1
    mask = np.zeros((yperc2.shape[1],), dtype=bool)
    mask[zeroind] = True
    yperc2[:, mask] = ~(np.any(yperc2[:, ~mask], axis=1).reshape((-1,1)))
    return yperc, yperc2


def blockshapedy(arr, y, tilesize, zeroind=4, steps_per=2, return_class=False, n_dim=4):

    tiles,y_rep = blockshapedy_repeat(arr, y, tilesize, steps_per=steps_per)
    yperc, yperc2 =  blockshapedy_transform(tiles, y_rep, n_dim=n_dim, zeroind=zeroind)
    #yperc = np.reshape(tiles, (len(tiles),-1)).mean(axis=1).reshape(-1,1)
    if return_class:
        return yperc, yperc2
    else:
        return yperc


def blockshaped_location(arr,tilesize,steps_per=2, n_dim=4, n_jobs=3):

    tiles =blockshaped_location_target(arr, tilesize, steps_per=steps_per, n_jobs=n_jobs)
    out_tiles =  blockshaped_location_transform(tiles, n_dim=n_dim, n_jobs=n_jobs)

    return out_tiles

def blockshaped_location_target(arr, tilesize, steps_per=2, n_jobs=3):
    #tiles = Parallel(n_jobs=n_jobs)(
    #    delayed(extract_patches)(arr[ind], tilesize, steps_per) for ind in range(len(arr)))
    tiles = [extract_patches(arr[ind], tilesize, steps_per) for ind in range(len(arr))]

    tiles = np.concatenate(tiles, axis=0)
    return tiles

def blockshaped_location_transform(tiles, n_dim=4, n_jobs=3):

    targets = Parallel(n_jobs=n_jobs)(delayed(tile_targets)(tile, n_dim=n_dim) for tile in tiles)

    yperc = np.concatenate(targets, axis=0)
    return yperc

class BlockShapedLocationTransform(object):

    def __init__(self, n_dim=4):
        self.n_dim = n_dim

    def __call__(self, y_vals):
        return blockshaped_location_transform(y_vals,n_dim=self.n_dim)


def reconstruct_patches(patches, coords, shape, tilesize, flat=False):
    if flat:
        if io_utils.BACKEND=='th':
            patches = patches.reshape((-1, 1,tilesize,tilesize))
        else:
            patches = patches.reshape((-1,tilesize, tilesize, 1))

    if io_utils.BACKEND=='th':
        dim = patches.shape[1]
        out = np.zeros((dim,)+shape)
        out_count = np.zeros((dim,) +shape)
    else:
        dim = patches.shape[-1]
        out = np.zeros(shape+ (dim,))
        out_count = np.zeros(shape+(dim,))

    row_offset = 0
    for coord, patch in zip(coords, patches):
        row_start, col_start = coord
        if io_utils.BACKEND == 'th':
            out[:,row_start:row_start+tilesize,col_start:col_start+tilesize] += patches[row*nper[1]+col]
            out_count[:,row_start:row_start+tilesize,col_start:col_start+tilesize] +=1
        else:
            out[row_start:row_start + tilesize, col_start:col_start + tilesize, :] += patches[row * nper[1] + col]
            out_count[row_start:row_start + tilesize, col_start:col_start + tilesize, :] += 1


    return out/out_count