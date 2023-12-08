#average the 6 subjects including missing y genes for female donor15496
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from vast import surface_tools
import pandas as pd
import scipy.stats as stats
import subprocess
import paths as p

base_dir = p.allen_dir
subjects=['donor10021','donor12876','donor14380','donor15496','donor15697','donor9861']
cortex=nib.load(os.path.join(p.fs_LR32k_dir,'Glasser_2016.32k.L.label.gii'))
cortex=cortex.darrays[0].data>0
version='_2'
n_genes=20781
hemi_c='L'
all_subjects=[]
for subject in subjects:
    print('loading {}'.format(subject))
    all_zs=[]
    chunks=10
    interval=np.ceil(n_genes/chunks).astype(int)
    for k in range(chunks):
        
        zs=nib.load(os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k',
                                             '{}.{}.z_score_expression_genes_{}_{}.func.gii'.format(subject,hemi_c,interval*k,
                                                                                                np.min([interval*(k+1),n_genes]))))
        all_zs.append(np.array(zs.agg_data()))
        #break
    all_zs=np.hstack(all_zs)
    all_subjects.append(all_zs)
#all_subjects=np.vstack(all_subjects)

#deal with y chromosome
chromosomes = np.loadtxt(os.path.join(p.phate_dir,'data','chromosome_allen.txt', dtype='str')
y=chromosomes=='Y'
not_y=chromosomes!='Y'

concat = all_subjects[0]+all_subjects[1]+all_subjects[2]+all_subjects[3]+all_subjects[4]+all_subjects[5]
concat[:,y] = concat[:,y] - all_subjects[3][:,y]
concat[:,not_y]=concat[:,not_y]/6
concat[:,y]=concat[:,y]/5

np.save(os.path.join(p.hcp_dir,f'all_subs_smoothed_z.npy'),concat.T)
