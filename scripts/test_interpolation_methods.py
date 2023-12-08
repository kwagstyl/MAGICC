#implementation of a number of interpolatation methods evaluated 

import nibabel as nib
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import os

from vast import surface_tools
import pandas as pd
import scipy.stats as stats
import stripy as stripy
import subprocess
import time
import pykrige
import torch
import gpytorch
import sys
sys.path.append('../scripts')
import nb_loaders as nbl

gene_dir= '/data1/bigbrain/phate_testing/armin_data/'

def map_samples_to_indices(subject,subjects_dir,fs_hemi):
    #set up list of hemispheres
    hemis=['lh','rh']
    hemis_c=['L','R']
    hemi_c=hemis_c[hemis.index(fs_hemi)]
    print('tidying cortex labels for {}'.format(subject))
    #load surface for neighbours
    mid_surf = nib.freesurfer.read_geometry(os.path.join(subjects_dir,subject, 'surf/{}.mid'.format(fs_hemi)))
    #get cortex vertices and make into a mask
    cort = np.sort(nib.freesurfer.read_label(os.path.join(subjects_dir, subject,'label/{}.cortex.label'.format(fs_hemi))))
    cortex_mask=np.ones(len(mid_surf[0]))
    cortex_mask[cort]=0
    #get neighbours
    neighbours=surface_tools.get_neighbours_from_tris(mid_surf[1])
    #tidy cortex mask which had holes.
    new_mask=surface_tools.tidy_holes_binary(cortex_mask,neighbours,threshold_area=5000)
    #invert
    #new_mask=new_mask*-1+1
    print('{:.0f} vertices changed'.format(np.sum(np.abs(cortex_mask-new_mask))))
    new_cortex_label=np.where(new_mask==0)[0]
    #load in dataframe of where samples are
    sample_location_df=pd.read_csv(os.path.join(gene_dir,'{}_SampleLocations_{}.csv'.format(subject,hemi_c)))
    sample_coordinates=np.array([sample_location_df['mri_x'],sample_location_df['mri_y'],sample_location_df['mri_z']]).T
    #find vertices only on the cortex. Careful, because there are sometimes duplicates
    sample_indices=surface_tools.get_nearest_indices(sample_coordinates,mid_surf[0][new_mask==0])
    sample_indices=np.where(new_mask==0)[0][sample_indices]
    vecs=sample_coordinates-mid_surf[0][sample_indices]
    #some samples are 30mm from cortex. We could consider filtering these out.
    distances=np.linalg.norm(vecs,axis=1)
    if len(sample_indices) != len(np.unique(sample_indices)):
        print("Warning: Multiple samples assigned to same vertex.")
    return sample_indices #distances #, new_cortex_label



def find_nearest_lats_lons(lats,lons,sample_vertices, lats_fslr,lons_fslr):
    """find nearest vertex based on lat and lon of old vertex"""
    fslr_vertices=np.zeros(len(sample_vertices)).astype(int)
    for k,vertex in enumerate(sample_vertices):
        orig_lat=lats[vertex]
        orig_lon = lons[vertex]
        fslr_vertices[k]=np.argmin(np.linalg.norm(np.array([lats_fslr,lons_fslr]).T-np.array([orig_lat,orig_lon]), axis=1, ord=2))
    return fslr_vertices

def get_fslr_samples(subject, base_dir, hemi):
    """ get indices of samples on fslr and dictionary indices, which accounts for duplicates """
    
    hemis=['L','R']
    hemis_fs=['lh','rh']
    hemi_c=hemis[hemis_fs.index(hemi)]
    #get sample indices in native space
    sample_indices=map_samples_to_indices(subject,base_dir,hemi)
    #get unique indices. TODO Duplicates should theoretically be averaged or something
    #dictionary indices maps unique samples back to their original rows in the matrix
    unique_samples,dictionary_indices=np.unique(sample_indices,return_inverse=True)
    #Import spherical coordinates for interpolation step
    sphere = nib.freesurfer.read_geometry(os.path.join(base_dir,subject, 'surf/{}.sphere'.format(hemi)))
    spherical_coords= surface_tools.spherical_np(sphere[0])
    lats, lons = spherical_coords[:,1]-np.pi/2, (spherical_coords[:,2])%(2*np.pi)
    #save latitudes and longitudes in native space
    gifti_demo=nib.load(os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/data_{}.func.gii'.format(subject,hemi_c)))
    gifti_demo.remove_gifti_data_array(0)
    gifti_demo.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(lats.astype(np.float32)))
    nib.save(gifti_demo,os.path.join(base_dir, subject,'surf','{}.lats.dscalar.gii'.format(hemi)))
    gifti_demo.remove_gifti_data_array(0)
    gifti_demo.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(lons.astype(np.float32)))
    nib.save(gifti_demo,os.path.join(base_dir, subject,'surf','{}.lons.dscalar.gii'.format(hemi)))
    os.environ["SUBJECTS_DIR"]= base_dir
    print('registering')
    #map to fs_LR 32K using workbench. 
    subprocess.call('wb_command -metric-resample {} {} {} ADAP_BARY_AREA {} -area-surfs {} {}'.format(
        os.path.join(base_dir,subject,'surf','{}.lats.dscalar.gii'.format(hemi)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/Native/{}.{}.sphere.reg.native.surf.gii'.format(subject,subject,hemi_c)),
           os.path.join(base_dir,'hcp_surfs','standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.{}.sphere.32k_fs_LR.surf.gii'.format(hemi_c)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.lats.dscalar.gii'.format(subject,subject,hemi_c)), 
            os.path.join(base_dir,'hcp_surfs','{}/T1w/Native/{}.{}.midthickness.native.surf.gii'.format(subject,subject,hemi_c)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.midthickness.32k_fs_LR.surf.gii'.format(subject,subject,hemi_c))),shell=True)
    subprocess.call('wb_command -metric-resample {} {} {} ADAP_BARY_AREA {} -area-surfs {} {}'.format(
        os.path.join(base_dir,subject,'surf','{}.lons.dscalar.gii'.format(hemi)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/Native/{}.{}.sphere.reg.native.surf.gii'.format(subject,subject,hemi_c)),
           os.path.join(base_dir,'hcp_surfs','standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.{}.sphere.32k_fs_LR.surf.gii'.format(hemi_c)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.lons.dscalar.gii'.format(subject,subject,hemi_c)), 
            os.path.join(base_dir,'hcp_surfs','{}/T1w/Native/{}.{}.midthickness.native.surf.gii'.format(subject,subject,hemi_c)),
            os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.midthickness.32k_fs_LR.surf.gii'.format(subject,subject,hemi_c))),shell=True)
    #import coregistered latitudes from native surface
    lats_fslr = nib.load(os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.lats.dscalar.gii'.format(subject,subject,hemi_c)))
    lons_fslr = nib.load(os.path.join(base_dir,'hcp_surfs','{}/MNINonLinear/fsaverage_LR32k/{}.{}.lons.dscalar.gii'.format(subject,subject,hemi_c)))
    #Use these to find positions of original samples
    fslr_vertices=find_nearest_lats_lons(lats,lons,unique_samples, lats_fslr.darrays[0].data,lons_fslr.darrays[0].data)
    return fslr_vertices, dictionary_indices

def get_fslr_rowsofexpr(subject, base_dir, fslr_vertices, dictionary_indices):
    """ get spherical coords of fs_LR 
    unique_vertices_fslr - unique vertices on the fslr surface 
    expr_row_dict gene expression row"""
    expr_row_dict={}
    unique_vertices_fslr=np.unique(fslr_vertices)
    inds=np.arange(len(fslr_vertices))
    inds2=np.arange(len(dictionary_indices))
    sphere=surface_tools.io_mesh.load_mesh_geometry(os.path.join(base_dir,'hcp_surfs',subject,
                                                                 'MNINonLinear','fsaverage_LR32k',
                                    '{}.L.sphere.32k_fs_LR.surf.gii'.format(subject)))
    spherical_coords= surface_tools.spherical_np(sphere['coords'])
    for unique_vertex_fslr in unique_vertices_fslr:
        expr_row_dict[unique_vertex_fslr]=[]
        dict_rows=inds[fslr_vertices==unique_vertex_fslr].tolist()
        for row in dict_rows:
            expr_row_dict[unique_vertex_fslr].extend(inds2[dictionary_indices==row].tolist())

    return  lats_sph, lons_sph, unique_vertices_fslr, expr_row_dict

def get_samples_and_vals(lats_sph, lons_sph, unique_vertices_fslr, expr_row_dict, genes_samples,
                         gene_index):
    """sample values on spherical space"""
    hemi_c='L'
    lats_t=[]
    lons_t=[]
    vals_t=[]
    for unique_vertex_fslr in unique_vertices_fslr:
        lats_t.append(lats_sph[unique_vertex_fslr])
        lons_t.append(lons_sph[unique_vertex_fslr])
        #At this point, means only and not weighted. Might consider weighting
        vals_t.append(np.mean(genes_samples[np.array(expr_row_dict[unique_vertex_fslr]),gene_index]))

    return np.array(lats_t), np.array(lons_t), np.array(vals_t)

def nearest_neighbour_smoother(lats_t, lons_t, vals_t, lats_sph, lons_sph):
    """nearest neighbour smoothing interpolation
    lats_t, lons_t are the sample coordinates
    vals_t are the values at the sample coordinates
    lats_sph, lons_sph are the coordinates of the vertices on the sphere"""
    base_dir = '/data1/allen_surfaces'
    mesh=stripy.sTriangulation(lons_t,lats_t)
    #linear interpolation
    interpolated=mesh.interpolate(lons_sph,lats_sph,vals_t,order=0)[0]
    gifti_demo=nib.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k/fs_LR32k.L.K1.func.gii'))
    gifti_demo.remove_gifti_data_array(0)
        #replace data with gene vertex array
    gifti_demo.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(interpolated.astype(np.float32)))
    nib.save(gifti_demo,os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'raw.func.gii'))
    #smoothing step
    print('smoothing')
    subprocess.call('wb_command -metric-smoothing {} {} 20 {}'.format(
            os.path.join(base_dir,'hcp_surfs','fs_LR32k','fs_LR.32k.L.inflated.surf.gii'),
                os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'raw.func.gii'),
                            os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'smoothed.func.gii')),
                       shell=True)

    #load smoothed data
    smoothed=nib.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'smoothed.func.gii'))
        
    vertex_vals=np.array(smoothed.agg_data())  
    return vertex_vals


def nearest_neighbour_smoother_arr(lats_arr, lons_arr, vals_arr, lats_sph, lons_sph):
    """nearest neighbour smoothing interpolation
    lats_arr, lons_arr are the sample coordinates
    vals_arr are the values at the sample coordinates
    lats_sph, lons_sph are the coordinates of the vertices on the sphere"""
    base_dir = '/data1/allen_surfaces'
    interpolated = np.zeros((len(lats_arr),len(lats_sph),))
    for sub in np.arange(len(lats_arr)):
        lats_t=lats_arr[sub]
        lons_t=lons_arr[sub]
        vals_t=vals_arr[sub]
        mesh=stripy.sTriangulation(lons_t,lats_t)
        #linear interpolation
        interpolated[sub]=mesh.interpolate(lons_sph,lats_sph,vals_t,order=0)[0]

    gifti_demo=nb.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k/fs_LR32k.L.K1.func.gii'))
    gifti_demo.remove_gifti_data_array(0)
        #replace data with gene vertex array
    gifti_demo.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(interpolated.astype(np.float32).T))
    nb.save(gifti_demo,os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'raw.func.gii'))
    #smoothing step
    print('smoothing')
    subprocess.call('wb_command -metric-smoothing {} {} 20 {}'.format(
            os.path.join(base_dir,'hcp_surfs','fs_LR32k','fs_LR.32k.L.inflated.surf.gii'),
                os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'raw.func.gii'),
                            os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'smoothed.func.gii')),
                       shell=True)

    #load smoothed data
    smoothed=nb.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'smoothed.func.gii'))
        
    vertex_vals=np.array(smoothed.agg_data())  
    return vertex_vals


def _get_weights(dist):
    """ Gets inverse of `dist`, handling potential infs

    Parameters
    ----------
    dist : array_like
        Distances to be converted to weights

    Returns
    -------
    weights : np.ndarray
        Inverse of `dist`
    """

    with np.errstate(divide='ignore'):
        dist = 1. / dist
    isinf = np.isinf(dist)
    infrow = np.any(isinf, axis=1)
    dist[infrow] = isinf[infrow]

    return dist

def abagen_interpolator(coords,sample_coords,expr,n_neighbours=10,timeit=False):
    """implement method for abagen-style interpolation
    coords are the coordinates of the vertices on the sphere
    sample_coords are the sample coordinates
    expr are the expression values at the sample coordinates
    """
    from scipy.spatial import KDTree
    if timeit:
        t1=time.time()
    tree= KDTree(np.c_[sample_coords[:,0], sample_coords[:,1],
                       sample_coords[:,2]])
    dist, idx = tree.query(coords, k=n_neighbours)
    dist = _get_weights(dist)

    # get average of nearest neighbors
    dense = np.zeros((expr.shape[0],len(coords)))
    denom = np.sum(dist, axis=1)
    for n, j in enumerate(expr):
        num = np.sum(j[idx] * dist, axis=1)
        dense[n] = num / denom
    if timeit:
        t2=time.time()
        return dense, t2-t1
    return dense

def kriging(lats_t, lons_t, vals_t, lats_sph, lons_sph,
            variogram=None,model='power', timeit=False):
    """2d kriging
    lats_t, lons_t are the sample coordinates
    vals_t are the values at the sample coordinates
    lats_sph, lons_sph are the coordinates of the vertices on the sphere
    variogram is a dictionary of variogram parameters
    model is the variogram model
    timeit is a boolean to return time taken"""
    if timeit:
        t1=time.time()
    kriger=pykrige.OrdinaryKriging(np.rad2deg(lons_t)+180,np.rad2deg(lats_t),vals_t,
                                variogram_model=model,
                               #variogram_parameters= None,
                               variogram_parameters= variogram,
                                #variogram_parameters={'sill': s, 'range': r, 'nugget': n},
                               coordinates_type='geographic', exact_values=False)
    vertex_vals=kriger.execute('points',np.rad2deg(lons_sph)+180,np.rad2deg(lats_sph))[0]
    if timeit:
        t2=time.time()
        return vertex_vals, t2-t1

    return vertex_vals


def kriging3d(sample_coords, vals_t, coords,
               variogram=None,model='power', timeit=False):
    """3d kriging
    sample_coords are the sample coordinates
    vals_t are the values at the sample coordinates
    lats_sph, lons_sph are the coordinates of the vertices on the sphere
    variogram is a dictionary of variogram parameters
    model is the variogram model
    timeit is a boolean to return time taken"""
    if timeit:
        t1=time.time()
    kriger=pykrige.OrdinaryKriging3D(sample_coords[:,0],sample_coords[:,1],sample_coords[:,2],vals_t,
                                variogram_model=model,
                               variogram_parameters= variogram,
                                exact_values=False)
    vertex_vals = kriger.execute('points',coords[:,0],coords[:,1],coords[:,2])[0]
    if timeit:
        t2=time.time()
        return vertex_vals, t2-t1
    return vertex_vals
#
def stripy_smoothing(lats_t, lons_t, vals_t, lats_sph, lons_sph):
    """implement method for stripy smooothing"""
    lats_grid=np.linspace(-np.pi/2,np.pi/2,500)
    lons_grid=np.linspace(-np.pi,np.pi,500)
    la,lo=np.meshgrid(lats_grid,lons_grid)
    la=la.ravel()
    lo=lo.ravel()
    mesh_g = stripy.sTriangulation(lo,la,permute=True)
    mesh2=stripy.sTriangulation(lons_sph,lats_sph)
    fs=mesh_g.interpolate(lons_sph, lats_sph, ff.T.ravel(),order=1)
    ave, norm, count = weighted_average_to_nodes(lons_t,lats_t, np.array(vals_t), mesh2)
    unconstrained_locations = np.where(norm < 1.0e-5)[0]
    ave[unconstrained_locations]=interpolated_n[0][unconstrained_locations]
    #lower p2 means smoothed
    p2=0.000001
    predictor_smooth = compute_smoothed_solutions(mesh2, norm, 0.5, ave, p2)
    return vertex_vals

def stripy_tension_splines(lats_t, lons_t, vals_t, lats_sph, lons_sph):
    """TODO implement method for stripy tension splines"""
    return vertex_vals

#if __name__ == "__main__":
 #   gene_dir= '/data1/bigbrain/phate_testing/allen_sample_data/AllenRegion2Region/'
  #  genes_samples=pd.read_csv(os.path.join(gene_dir,'{}_Expression_{}.csv'.format(subject,hemi_c)))
   # genes_samples=np.array(genes_samples)    




def calculate_gradient(expr_map,fwhm=5):
    base_dir = '/data1/allen_surfaces'
    gifti_demo=nb.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k/fs_LR32k.L.K1.func.gii'))
    gifti_demo.remove_gifti_data_array(0)
        #replace data with gene vertex array
    gifti_demo.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(expr_map.astype(np.float32)))
    input = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr.func.gii')
    output_mags = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr_mags.func.gii')
    output_grads = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr_grads.func.gii')
    surf_f = os.path.join(base_dir,'hcp_surfs','fs_LR32k','fs_LR.32k.L.very_inflated.surf.gii')
    nb.save(gifti_demo,input)
    if fwhm>0:
        subprocess.call('wb_command -metric-gradient {} {} {} -presmooth {}'.format(
                          surf_f,
                            input,
                            output_mags,
                           fwhm),
                           shell=True)
    else: 
        subprocess.call('wb_command -metric-gradient {} {} {}'.format(
                          surf_f,
                            input,
                            output_mags),
                           shell=True)
    mags=nb.load(output_mags)
    return np.vstack(mags.agg_data())

def calculate_gradient_vectors(expr_map,fwhm=5):
    base_dir = '/data1/allen_surfaces'
    gifti_demo=nb.load(os.path.join(base_dir,'hcp_surfs','fs_LR32k/fs_LR32k.L.K1.func.gii'))
    gifti_demo.remove_gifti_data_array(0)
        #replace data with gene vertex array
    gifti_demo.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(expr_map.astype(np.float32)))
    input = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr.func.gii')
    output_mags = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr_mags.func.gii')
    output_grads = os.path.join(base_dir,'hcp_surfs','fs_LR32k',
                                         'expr_grads.func.gii')
    surf_f = os.path.join(base_dir,'hcp_surfs','fs_LR32k','fs_LR.32k.L.flat.surf.gii')
    nb.save(gifti_demo,input)
    if fwhm>0:
        subprocess.call('wb_command -metric-gradient -presmooth {} {} {} {} -vectors {}'.format(
                        fwhm,
                        surf_f,
                            input,
                            output_mags,
                            output_grads),
                            shell=True)
    else: 
        subprocess.call('wb_command -metric-gradient  {} {} {} -vectors {}'.format(
                            surf_f,
                            input,
                            output_mags,
                            output_grads),
                            shell=True)
    mags=nb.load(output_mags)
    grads = nbl.load_vectors(output_grads)
    return np.vstack(mags.agg_data()), grads


def gp_interpolation(known_xyz, known_values, new_xyz, num_iterations=150, kernel=gpytorch.kernels.RBFKernel()):
    """
    Perform GP regression with known data and estimate values at new coordinates.

    Arguments:
    - known_xyz: array-like, shape (n_samples, 3)
        Known x, y, z coordinates.
    - known_values: array-like, shape (n_samples, n_features)
        Known expression values corresponding to the known coordinates.
    - new_xyz: array-like, shape (n_estimation_points, 3)
        New x, y, z coordinates to estimate values.

    Returns:
    - new_values: array-like, shape (n_estimation_points, n_features)
        Estimated values at the new coordinates.
    """

    # Min-max normalization for inputs
    known_values=np.array(known_values)
    known_xyz=np.array(known_xyz)
    new_xyz=np.array(new_xyz)
    
    known_xyz_min = known_xyz.min(axis=0)
    known_xyz_max = known_xyz.max(axis=0)
    known_values_min = known_values.min(axis=0)
    known_values_max = known_values.max(axis=0)

    known_xyz_norm = (known_xyz - known_xyz_min) / (known_xyz_max - known_xyz_min)
    known_values_norm = (known_values - known_values_min) / (known_values_max - known_values_min)
    new_xyz_norm = (new_xyz - known_xyz_min) / (known_xyz_max - known_xyz_min)

    # Convert data to tensors and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_xyz_norm = torch.tensor(known_xyz_norm, dtype=torch.float32, device=device)
    known_values_norm = torch.tensor(known_values_norm, dtype=torch.float32, device=device)
    new_xyz_norm = torch.tensor(new_xyz_norm, dtype=torch.float32, device=device)

    # Set up the Gaussian Process model and move to GPU
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel=gpytorch.kernels.RBFKernel()):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Create the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(known_xyz_norm, known_values_norm, likelihood, kernel=kernel).to(device)
    # Set the model to training mode and optimize hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    for _ in range(num_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(known_xyz_norm)
        # Calculate loss
        loss = -likelihood(output, known_values_norm).log_prob(known_values_norm)
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
    # Set the model to evaluation mode
    model.eval()
    likelihood.eval()
    # Predict values at new coordinates
    with torch.no_grad():
        observed_pred = likelihood(model(new_xyz_norm))
    # Get predicted mean values
    new_values_norm = observed_pred.mean.detach().cpu().numpy()

    # Re-expand the normalized values
    new_values = (new_values_norm * (known_values_max - known_values_min)) + known_values_min

    return new_values.astype(np.float16)




def interpolate(features,sample_indices,
                method='kriging',
                cortex=None,
                coords=None,
                num_iterations=150,
                kernel=gpytorch.kernels.RBFKernel()):
    """Interpolate features using a given method.
    features: full feature matrix
    sample_indices: indices of samples to use for interpolation
    method: interpolation method to use
    cortex: boolean array of vertices in the cortex
    coords: coordinates of vertices in the surface mesh - """
    #interpolate the data
    interpolated_data = np.zeros((len(features),coords.shape[0]))
    n_subs = len(sample_indices)
    spherical_coords= surface_tools.spherical_np(coords)
    lats, lons = spherical_coords[:,1]-np.pi/2, spherical_coords[:,2]
    for fi,feature in enumerate(features):
        sample_coords = []
        sample_values = []
        sample_lats = []
        sample_lons = []
        for subject_i in np.arange(n_subs):
            samples = sample_indices[subject_i]
            sample_coords.append(coords[samples])
            sample_values.append(feature[subject_i,samples])
            sample_lats.append(lats[samples])
            sample_lons.append(lons[samples])
        
        if method =='nearest_neighbour':
            interp_nn = nearest_neighbour_smoother_arr(sample_lats, sample_lons, 
                                                                      sample_values, lats, lons)
            z_scored = (interp_nn.T - np.mean(interp_nn[:,cortex],axis=1))/np.std(interp_nn[:,cortex],axis=1)
            interpolated_data[fi] = z_scored.mean(axis=1)
        sample_coords = np.concatenate(sample_coords)
        sample_values = np.concatenate(sample_values)
        sample_lats = np.concatenate(sample_lats)
        sample_lons = np.concatenate(sample_lons)
        if method =='kriging':
            interpolated_data[fi] = kriging(sample_lats, sample_lons, sample_values, lats, lons)
        elif 'kriging3d' in method:
            interpolated_data[fi] = kriging3d(sample_coords, sample_values, coords)
        elif 'gp' in method:
            interpolated_data[fi] = gp_interpolation(sample_coords, sample_values, coords,
                                                     num_iterations=num_iterations,
                                                     kernel=kernel)
        elif 'knn' in method: 
            interpolated_data[fi] = abagen_interpolator(coords,sample_coords, np.array([sample_values]), )
        
    return interpolated_data

