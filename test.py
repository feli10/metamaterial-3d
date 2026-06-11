import numpy as np
from Data_Generation.strut_functions import make_strut, combine_struts, make_frame
from Data_Generation.view_cell import view_cell 
from Property_Testing.homopy_fem_3d import homogenize_3d, IsotropicMaterial

# cell_size = 10
# cell = make_strut((0,0,0), (2,2,2), cell_size, 2)
# frame = make_frame(cell_size, 1)
# test = combine_struts([frame, cell])
# view_cell(test, title=f"size: {cell_size}")

mat  = IsotropicMaterial(E=1.0, nu=0.3)

d = np.load("cells.npz")
cells  = d["cells"]
labels = d["labels"]

print(cells.shape)
print(labels.shape)

for i in range(1290,1297):
    view_cell(cells[i], title=f"cell {i} | thickness {labels[i]}")
