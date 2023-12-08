# Description: Create spin permutations for the HCP surface
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import os
from scipy.stats import special_ortho_group
import time
from  nilearn import plotting 
from scipy.spatial import cKDTree
base_dir = '/data1/allen_surfaces/'
spin_dir = '/data1/bigbrain/phate_testing/spin_dir/'

sphere = nb.load('/data1/allen_surfaces/hcp_surfs/standard_mesh_atlases/L.sphere.32k_fs_LR.surf.gii')
cortex=nb.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k','Glasser_2016.32k.L.label.gii'))
cortex=cortex.darrays[0].data>0

coords=sphere.darrays[0].data
coords_cortex=coords[cortex]
n_perm=10000

tree=cKDTree(coords_cortex)
indices=np.zeros((n_perm,len(coords_cortex))).astype(int)
for x in np.arange(n_perm):
    rotation = special_ortho_group.rvs(3)
    new_coords = coords_cortex @ rotation
    distance, indices[x]=tree.query(new_coords,k=1)
    
np.save(os.path.join(spin_dir,'spins_{}.npy'.format(n_perm)),indices)