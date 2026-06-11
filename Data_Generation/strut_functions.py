import numpy as np
from scipy import signal

def make_strut(start, end, cell_size, thickness):
    """
    :param start: [tuple] - the (x, y, z) coordinates of the strut's starting point, each in {0, 1, 2}
    :param end: [tuple] - the (x, y, z) coordinates of the strut's ending point, each in {0, 1, 2}
    :param cell_size: [int] - the size of the cubic cell that will be produced
    :param thickness: [int] - the number of pixels to be activated surrounding the base shape
    :return: [ndarray] - the output is a cubic unit cell with a single strut connecting the start and end nodes, expanded to the desired thickness. The activated voxels are 1 and the deactivated voxels are 0
    """
    start = np.round(np.array(start) * (cell_size - 1) / 2).astype(int)
    end = np.round(np.array(end)   * (cell_size - 1) / 2).astype(int)
    
    n = np.abs(end - start).max() + 1
    xs = np.linspace(start[0], end[0], n).astype(int)
    ys = np.linspace(start[1], end[1], n).astype(int)
    zs = np.linspace(start[2], end[2], n).astype(int)

    A = np.zeros((int(cell_size), int(cell_size), int(cell_size)))
    A[zs, ys, xs] = 1

    A = add_thickness(A, thickness)
    return A

def ball_kernel(radius):
    """
    :param radius: [float] - voxels within this distance of the center are activated
    :return: [ndarray] - a cubic ball-shaped kernel, 1 inside the radius and 0 outside
    """
    size = 2 * int(np.ceil(radius)) + 1
    c = size // 2
    zz, yy, xx = np.ogrid[:size, :size, :size]
    dist = np.sqrt((zz - c)**2 + (yy - c)**2 + (xx - c)**2)
    return (dist <= radius).astype(float)

def add_thickness(array_original, thickness):
    """
    :param array_original: [ndarray] - a strut array with thickness 1, of any shape
    :param thickness: [int] - 0 removes the strut, 1 keeps the thinnest 1-voxel strut,
    higher values thicken it with a solid ball kernel
    :return: [ndarray] - the thickened strut. Activated voxels are 1, deactivated are 0
    """
    A = array_original
    if thickness <= 0:
        A[A > 0] = 0
    else:
        radius = thickness - 0.5
        kernel = ball_kernel(radius)
        convolution = signal.convolve(A, kernel, mode='same', method='direct')
        A = np.where(convolution >= 1, 1, 0)
    return A

def combine_struts(arrays):
    """
    :param arrays: [list] - a list of strut arrays to be combined into one cell
    :return: [ndarray] - a cell of shape (z, y, x) containing the union of all input structs, with activated voxels as 1 and empty voxels as 0
    """
    output_array = np.sum(arrays, axis=0) 
    output_array = np.array(output_array > 0, dtype=int) 
    return output_array

def make_frame(cell_size, thickness):
    node_pairs = [
        ((0,0,0),(0,0,2)), ((0,0,0),(0,2,0)), ((0,0,0),(2,0,0)),
        ((0,0,2),(0,2,2)), ((0,0,2),(2,0,2)), ((0,2,0),(0,2,2)),
        ((0,2,0),(2,2,0)), ((0,2,2),(2,2,2)), ((2,0,0),(2,0,2)),
        ((2,0,0),(2,2,0)), ((2,0,2),(2,2,2)), ((2,2,0),(2,2,2))
    ]

    struts = [make_strut(a, b, cell_size, thickness) for (a, b) in node_pairs]
    return combine_struts(struts)

