import numpy as np
from Data_Generation.strut_functions import make_strut, combine_struts, make_frame
from Data_Generation.view_cell import view_cell 
from Property_Testing.homopy_fem_3d import homogenize_3d, IsotropicMaterial

for cell_size in [10, 15, 20]:
    frame = make_frame(cell_size, 1.5)
    view_cell(frame, title=f"size: {cell_size}, thickness: 1.5")

# mat  = IsotropicMaterial(E=1.0, nu=0.3)
# cell = combine_struts([make_strut((1,1,0),(1,1,2), 10, 2)]) # thick vertical bar
# C = homogenize_3d(10, 10, 10, mat, density_field=cell)

# print(C.shape)
# print(np.round(C, 4))