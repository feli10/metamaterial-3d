import numpy as np
import matplotlib.pyplot as plt

def view_cell(A, threshold=0.0, title=None):
    """
    :param A: [ndarray] - a 3D unit cell of shape (z, y, x)
    :param threshold: [float] - the cutoff above which a voxel is rendered
    :param title: [str] - an optional title displayed above the plot
    :return: [None] - this function renders the cell and does not return a value
    """
    filled = A > threshold                      
    filled = np.transpose(filled, (2, 1, 0))    

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, edgecolor='k', linewidth=0.2)

    ax.set_xlim(0, A.shape[2])   # x = axis 2
    ax.set_ylim(0, A.shape[1])   # y = axis 1
    ax.set_zlim(0, A.shape[0])   # z = axis 0

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()