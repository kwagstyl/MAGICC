
#script to generate main dataset
import nibabel as nib
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib_surface_plotting as msp
from vast import surface_tools
import pandas as pd
import scipy.stats as stats
import stripy as stripy
import subprocess
import matplotlib
import paths as p

np.random.seed(0)
cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))


def names_to_lobe_indices(structure_names):
    " convert structure names to lobe index "
    colors=np.zeros(len(structure_names))
    for k,structure in enumerate(structure_names):
        if 'cingulate' in structure or 'paraterminal' in structure:
            colors[k]=3 #[1,1,0,1]
        elif 'frontal' in structure or 'rostral' in structure or 'orbital' in structure or 'rectus' in structure:
            colors[k]=1 #[1,0,0,1]
        elif 'temporal' in structure or 'fusiform' in structure or 'hippo' in structure or 'Heschl' in structure or 'transvers' in structure or 'polare' in structure:
            colors[k]=7 #[0,1,0,1]
        elif 'parietal' in structure or 'angular'  in structure or 'precuneus' in structure or 'supramarginal' in structure:
            colors[k]=8 #[0,0,1,1]
        elif 'occipital' in structure or 'lingual' in structure or 'cuneus' in structure:
            colors[k]=4 #[0.5,0.1,0.8,1]
        elif 'olfactory' in structure or 'piriform' in structure:
            colors[k] = 1 #[0.6,0,0,1]
        elif 'insula' in structure:
            colors[k] = 9 #[0.8, 0.6, 0.1,1]
        elif 'postcentral' in structure:
            colors[k] = 5 #[0,0.8,1,1]
        elif 'precentral' in structure:
            colors[k] = 6 # [0.9,0.5, 0.6, 1]
        elif 'paracentral' in structure:
            colors[k] = 6 #[0.9,0.5, 0.1,1]
        else :
            print(structure)
    return colors

def v_to_c(i, j, k,M,abc):
    """ Return X, Y, Z coordinates for i, j, k """
    return M.dot([i, j, k]) + abc 

def map_samples_to_indices(subject,subjects_dir,fs_hemi, lobe_filter=False):
    hemis=['lh','rh']
    hemis_c=['L','R']
    hemi_c=hemis_c[hemis.index(fs_hemi)]
    #print('tidying cortex labels for {}'.format(subject))
    #load surface for neighbours
    #read metadata or there is a shift!!!
    mid_surf = nib.freesurfer.read_geometry(os.path.join(subjects_dir,subject, 'surf/{}.mid'.format(fs_hemi)),
                  read_metadata=True)
    mid_surf = [np.array(mid_surf[0])+mid_surf[2]['cras'],mid_surf[1]]
    cortex_label = np.sort(nib.freesurfer.read_label(os.path.join(subjects_dir, subject,'label/{}.cortex.label'.format(fs_hemi))))
    cortex_mask=np.ones(len(mid_surf[0]))
    cortex_mask[cortex_label]=0
    
    sample_location_df=pd.read_csv(os.path.join(gene_dir,'{}_preprocessed'.format(subject),'{}_ctx_samples.csv'.format(subject)))
    #filter_hemisphere
    if hemi_c=='L':
        sample_location_df_hemis = sample_location_df[sample_location_df['mni_x']<0]
        hemi_mask=sample_location_df['mni_x']<0
    else:
        sample_location_df_hemis = sample_location_df[sample_location_df['mni_x']>=0]
        hemi_mask=sample_location_df['mni_x']>=0
    voxel_coordinates=np.array([sample_location_df_hemis['mri_voxel_x'],sample_location_df_hemis['mri_voxel_y'],sample_location_df_hemis['mri_voxel_z']]).T
    mri=nb.load(os.path.join(base_dir,subject,'mri','orig','001.mgz'))
    sample_coordinates=np.zeros_like(voxel_coordinates)
    M = mri.affine[:3, :3]
    abc = mri.affine[:3, 3]
    for r,c in enumerate(voxel_coordinates):
        sample_coordinates[r]=v_to_c(c[0],c[1],c[2],M,abc)
    #sample_coordinates=np.array([sample_location_df_hemis['mri_x'],sample_location_df_hemis['mri_y'],sample_location_df_hemis['mri_z']]).T
    #find vertices only on the cortex. Careful, because there are sometimes duplicates
    sample_indices=surface_tools.get_nearest_indices(sample_coordinates,mid_surf[0][cortex_mask==0])
    sample_indices=np.where(cortex_mask==0)[0][sample_indices]
    structure_names = np.array(sample_location_df_hemis['structure_name'])
    named_lobe_index = names_to_lobe_indices(structure_names)
    row_mask=np.ones_like(sample_indices).astype(bool)
    if lobe_filter:
        lobes=nib.freesurfer.read_annot(os.path.join(base_dir,subject,'label','{}.lobes.annot'.format(hemi)))
        sample_indices_lobes=np.zeros_like(sample_indices)
        for lobe_i in np.unique(named_lobe_index):
            sample_indices_lobe = surface_tools.get_nearest_indices(sample_coordinates[named_lobe_index==lobe_i],mid_surf[0][lobes[0]==lobe_i])
            sample_indices_lobes[named_lobe_index==lobe_i] = np.where(lobes[0]==lobe_i)[0][sample_indices_lobe]
        #sphere = nib.freesurfer.read_geometry(os.path.join(base_dir,subject, 'surf/{}.sphere'.format(hemi)))
        distances = np.linalg.norm(mid_surf[0][sample_indices]-mid_surf[0][sample_indices_lobes], axis=1)
        sample_indices = sample_indices[distances<20]
        structure_names = structure_names[distances<20]
        row_mask=distances<20
        #combine row and hemi mask
        hemi_mask[hemi_mask]=row_mask
    if len(sample_indices) != len(np.unique(sample_indices)):
        print("Warning: Multiple samples assigned to same vertex.")
    return sample_indices, structure_names,  hemi_mask,sample_coordinates




base_dir = p.allen_dir
gene_dir= os.path.join(p.phate_dir,'armin_data')


def names_to_colors(structure_names):
    " convert structure names to colors based on lobes "
    colors=np.zeros(len(structure_names))
    for k,structure in enumerate(structure_names):
        if 'cingulate' in structure or 'paraterminal' in structure:
            colors[k]=0 #[1,1,0,1]
        elif 'frontal' in structure or 'rostral' in structure or 'orbital' in structure or 'rectus' in structure:
            colors[k]=1 #[1,0,0,1]
        elif 'temporal' in structure or 'fusiform' in structure or 'hippo' in structure or 'Heschl' in structure or 'transvers' in structure or 'polare' in structure:
            colors[k]=2 #[0,1,0,1]
        elif 'parietal' in structure or 'angular'  in structure or 'precuneus' in structure or 'supramarginal' in structure:
            colors[k]=3 #[0,0,1,1]
        elif 'occipital' in structure or 'lingual' in structure or 'cuneus' in structure:
            colors[k]=4 #[0.5,0.1,0.8,1]
        elif 'olfactory' in structure or 'piriform' in structure:
            colors[k] = 5 #[0.6,0,0,1]
        elif 'insula' in structure:
            colors[k] = 6 #[0.8, 0.6, 0.1,1]
        elif 'postcentral' in structure:
            colors[k] = 7 #[0,0.8,1,1]
        elif 'precentral' in structure:
            colors[k] = 8 # [0.9,0.5, 0.6, 1]
        elif 'paracentral' in structure:
            colors[k] = 9 #[0.9,0.5, 0.1,1]
        else :
            print(structure)
    return colors


subjects={'R':['donor9861','donor10021'],
          'L':['donor12876','donor14380','donor15496','donor15697','donor9861','donor10021']}
lobe_filter=True
for k,hemi_c in enumerate(['L']): # ,'R']):
    dictionary_indices={}
    hemi=['lh','rh'][k]
    for subject in subjects[hemi_c]:
        #find indices of nearest vertices to surface
        sample_indices, structure_names,row_mask,sample_coordinates =map_samples_to_indices(subject,base_dir,hemi, lobe_filter=lobe_filter)
        #get unique indices.  
        unique_samples, dictionary_indices[subject]=np.unique(sample_indices,return_index=True)
        #Import spherical coordinates for interpolation step
        sphere = nib.freesurfer.read_geometry(os.path.join(base_dir,subject, 'surf/{}.sphere'.format(hemi)))
        spherical_coords= surface_tools.spherical_np(sphere[0])
        lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
        mesh=stripy.sTriangulation(lons[unique_samples],lats[unique_samples])
        #interpolate values, here this is nearest neighbour mapping to create annotation
        
        interpolated=mesh.interpolate(lons,lats,np.arange(len(unique_samples)),order=0)
        #PLOT THESE HERE
        vertices,faces = nib.freesurfer.read_geometry(os.path.join(base_dir,subject, 'surf/{}.white'.format(hemi)))
        structure_color = names_to_colors(structure_names)
        overlay_color  =    structure_color[dictionary_indices[subject][interpolated[0].astype(int)]]
#        msp.plot_surf(vertices,faces,overlay_color,filename='qc/{}_{}_native_interpolation_lobes.png'.format(subject,hemi_c),cmap='tab20',rotate=[90,270],label=True,
#          vmin=0,vmax=9)
#        msp.plot_surf(vertices,faces,interpolated[0],filename='qc/{}_{}_native_interpolation_indices.png'.format(subject,hemi_c),cmap=cmap,rotate=[90,270],label=True,
#          vmin=0,vmax=250)
        annotation=nib.freesurfer.read_annot(os.path.join(base_dir,subject,'label','{}.500.aparc.annot'.format(hemi)))
        cvals=np.vstack([255*cmap(np.unique(interpolated[0])/255).T,np.unique(interpolated[0])+1]).T.astype(int)
        cvals[:,3]=0
        #Save index locations as annotation to be mapped
        nib.freesurfer.write_annot(os.path.join(base_dir,subject,'label','{}.unique_samples.annot'.format(hemi)),
                                  interpolated[0].astype(int),cvals,
                                  names=dictionary_indices[subject].astype(str))
        os.environ["SUBJECTS_DIR"]= base_dir
        print('registering')
        #convert to gifti
        subprocess.call('mris_convert --annot {} {} {}'.format(
        os.path.join(base_dir,subject,'label','{}.unique_samples.annot'.format(hemi)),
        os.path.join(base_dir,subject,'surf','{}.white'.format(hemi)),
        os.path.join(base_dir,subject,'label','{}.unique_samples.label.gii'.format(hemi))),shell=True)

        #map to fs_LR 32K using workbench. 
        subprocess.call('wb_command -label-resample {} {} {} ADAP_BARY_AREA {} -area-surfs {} {}'.format(
            os.path.join(base_dir,subject,'label','{}.unique_samples.label.gii'.format(hemi)),
                os.path.join(p.hcp_dir,'{}/MNINonLinear/Native/{}.{}.sphere.reg.native.surf.gii'.format(subject,subject,hemi_c)),
               os.path.join(p.atlases_dir,'resample_fsaverage/fs_LR-deformed_to-fsaverage.{}.sphere.32k_fs_LR.surf.gii'.format(hemi_c)),
                os.path.join(p.hcp_dir,'{}/MNINonLinear/fsaverage_LR32k/{}.{}.unique_samples.label.gii'.format(subject,subject,hemi_c)), 
                os.path.join(p.hcp_dir,'{}/T1w/Native/{}.{}.midthickness.native.surf.gii'.format(subject,subject,hemi_c)),
                os.path.join(p.hcp_dir,'{}/MNINonLinear/fsaverage_LR32k/{}.{}.midthickness.32k_fs_LR.surf.gii'.format(subject,subject,hemi_c))),shell=True)

        
        #import sample data
        genes_samples=pd.read_csv(os.path.join(gene_dir,'{}_preprocessed'.format(subject),'{}_ctx_expr.csv'.format(subject)))
        #import annotation and extract mapped to fs_LR
        sample_annotation = nib.load(os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k','{}.{}.unique_samples.label.gii'.format(subject,hemi_c)))
        #PLOT INTERPOLATED AND IMPORTED FOR COMPARISON
        annot_data = sample_annotation.darrays[0].data
        overlay_color  =    structure_color[dictionary_indices[subject][annot_data]]
        fslr = nib.load(os.path.join(p.hcp_dir,'{}/MNINonLinear/fsaverage_LR32k/{}.{}.inflated.32k_fs_LR.surf.gii'.format(subject,subject,hemi_c)))
        #PLOT INTERPOLATED AND IMPORTED FOR COMPARISON
        glasser = nib.load(os.path.join(p.fs_LR32k_dir,'Glasser_2016.32k.L.label.gii'))
        glasser_data=glasser.darrays[0].data

        #make into array
        genes_samples=np.array(genes_samples).T[1:]
        genes_samples=genes_samples[row_mask]    
        #weird indexing because of uniques and nearest sample vertex finding up above. 
        #Indices of which samples are which vertex label stored above.
        vals_sampled=np.zeros((len(unique_samples),genes_samples.shape[1]))
        for k,sample in enumerate(unique_samples):
            vals_sampled[k]=np.mean(genes_samples[sample_indices==sample],axis=0)
        genes_mapped=vals_sampled[annot_data]
        print(genes_mapped.shape)
        #mask medial wall to zero
        genes_mapped[glasser_data==0]=0
        #create a split cifti to use for reading and writing
        subprocess.call('wb_command -cifti-separate {} COLUMN -metric CORTEX_LEFT {} -metric CORTEX_RIGHT {}'.format(
        os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k','{}.sulc.32k_fs_LR.dscalar.nii'.format(subject)),
        os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k','data_L.func.gii'),
        os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k','data_R.func.gii')
    ),shell=True)
        #split gene set into 3 for smoothing and z scoring (and gradients)
        #otherwise too slow
        chunks=10
        interval=np.ceil(genes_mapped.shape[1]/chunks).astype(int)
        for k in range(chunks):
            #read in demo gifti file
            gifti_demo=nib.load(os.path.join(p.hcp_dir,'{}/MNINonLinear/fsaverage_LR32k/data_{}.func.gii'.format(subject,hemi_c)))
            gifti_demo.remove_gifti_data_array(0)
            #replace data with gene vertex array
            gifti_demo.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(genes_mapped[:,interval*k:np.min([interval*(k+1),genes_mapped.shape[1]])].astype(np.float32)))
            #save
            raw_file = os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k',
                                             '{}.{}.raw_expression_genes_{}_{}.func.gii'.format(subject,hemi_c,interval*k,
                                                                                                np.min([interval*(k+1),genes_mapped.shape[1]])))
            nib.save(gifti_demo,raw_file)
            #smooth data step using workbench   
            print('smoothing step {} {}'.format(subject,k))
            smoothed_data_file = os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k',
                                             '{}.{}.smoothed_expression_genes_{}_{}.func.gii'.format(subject,hemi_c,interval*k,
                                                                                                np.min([interval*(k+1),genes_mapped.shape[1]])))
            subprocess.call('wb_command -metric-smoothing -fix-zeros {} {} 20 {}'.format(
                os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k','{}.{}.white.32k_fs_LR.surf.gii'.format(subject,hemi_c)),
                    raw_file,smoothed_data_file
                                ),
                           shell=True)

            #load smoothed data
            smoothed=nib.load(smoothed_data_file)

            smoothed_array=np.array(smoothed.agg_data())  
            #zscore data and then resave for later analysis
            z_scored=((smoothed_array.T-np.mean(smoothed_array,axis=1))/np.std(smoothed_array,axis=1))
            gifti_demo=nib.load(os.path.join(p.hcp_dir,'{}/MNINonLinear/fsaverage_LR32k/data_{}.func.gii'.format(subject,hemi_c)))
            gifti_demo.remove_gifti_data_array(0)
            gifti_demo.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(z_scored.astype(np.float32)))
            nib.save(gifti_demo,os.path.join(p.hcp_dir,subject,'MNINonLinear/fsaverage_LR32k',
                                             '{}.{}.z_score_expression_genes_{}_{}.func.gii'.format(subject,hemi_c,interval*k,
                                                                                                np.min([interval*(k+1),genes_mapped.shape[1]]))))
            os.remove(smoothed_data_file)
            os.remove(raw_file)
            

        
