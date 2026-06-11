import numpy as np
from Data_Generation.strut_functions import make_strut, combine_struts
from Data_Generation.view_cell import view_cell 
from Property_Testing.homopy_fem_3d import homogenize_3d, IsotropicMaterial

cell = combine_struts([
make_strut((1,1,0),(1,1,2),10,3), # vertical bar
make_strut((0,0,0),(2,2,2),10,2), # space diagonal
])
print(cell.shape)
view_cell(cell, title="bar + space diagonal")

# mat  = IsotropicMaterial(E=1.0, nu=0.3)
# cell = combine_struts([make_strut((1,1,0),(1,1,2), 10, 2)]) # thick vertical bar
# C = homogenize_3d(10, 10, 10, mat, density_field=cell)

# print(C.shape)
# print(np.round(C, 4))