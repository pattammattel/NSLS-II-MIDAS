from StackCalcs import *
from RefChooser import *
dff,col_name = create_df_from_nor(athenafile='marked2.nor')

elist = np.loadtxt('Site4um.txt')

#new_ref = interploate_E(dff.values,elist.values)

print((pd.DataFrame(elist)).values)
