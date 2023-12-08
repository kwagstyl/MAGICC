#create single subject interpolation atlases.

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from vast import surface_tools
import pandas as pd
import scipy.stats as stats
import subprocess
import nibabel as nb

base_dir = '/data1/allen_surfaces/'
subjects=['donor10021','donor12876','donor14380','donor15496','donor15697','donor9861']
cortex=nib.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k','Glasser_2016.32k.L.label.gii'))
cortex=cortex.darrays[0].data>0
hcp_dir = 'hcp_surfs'
n_genes=20781
hemi_c='L'
all_subjects=[]
for subject in subjects:
    print('loading {}'.format(subject))
    all_zs=[]
    chunks=10
    interval=np.ceil(n_genes/chunks).astype(int)
    for k in range(chunks):
        
        zs=nib.load(os.path.join(base_dir,hcp_dir,subject,'MNINonLinear/fsaverage_LR32k',
                                             '{}.{}.z_score_expression_genes_{}_{}.func.gii'.format(subject,hemi_c,interval*k,
                                                                                                np.min([interval*(k+1),n_genes]))))
        all_zs.append(np.array(zs.agg_data()))
        #break
    all_zs=np.hstack(all_zs)
    np.save('../subject_data/{}.npy'.format(subject),all_zs)