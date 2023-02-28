#class to download and load dataset
import subprocess
import os
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib_surface_plotting as msp

class MagiccDataset():
    def __init__(
    self,
    figshare = 'https://figshare.com/ndownloader/files/39434188?private_link=82c8f6ebda38af670cd1'):
        """
        """
        self.figshare = figshare
        self.download_data()
        self.load_data()
        self.gene_gradients=None


    def download_data(self):
        #check if file exists
        if not os.path.isdir('magicc_expression_data'):
            print('Downloading full dataset. This will take ~ 1-3 minutes')   
            subprocess.call(f'wget --content-disposition {self.figshare}',shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
            print('Unzipping ...')   
            subprocess.call(f'unzip magicc.zip',shell=True)
        return
    
    def load_data(self):
        print('loading in data')
        self.gene_expression = np.load('magicc_expression_data/ahba_vertex.npy')
        self.surf=nb.load(os.path.join( 'magicc_expression_data',
                                'fs_LR.32k.L.inflated.surf.gii'))
        self.parcellation = nb.load(os.path.join('magicc_expression_data',
                                            'Glasser_2016.32k.L.label.gii'
                                            ))
        self.cortex_mask = self.parcellation.darrays[0].data>0
        self.gene_info = pd.read_csv('magicc_expression_data/SuppTable2.csv')

    def plot_gene_expression(self,gene_name):
        """get index and plot a map of a gene name"""
        try:
            gene_index = np.where(self.gene_info['gene.symbol']==gene_name)[0][0]
            gene_reproducibility = self.gene_info['Estimated reproducibility'][gene_index]
        except : 
            print(f"{gene_name} not found. Please check spelling or for other aliases")
        msp.plot_surf(self.surf.darrays[0].data,self.surf.darrays[1].data,
                      self.gene_expression[gene_index],
                  rotate=[90,270],
                 cmap='turbo',vmin=-2,vmax=2,base_size=10,
              mask=~self.cortex_mask,
              mask_colour=np.array([0,0,0,1]),
                  colorbar=True,cmap_label='Z-scored\nexpression',
                  title = gene_name+f' expression \n Estimatated reproducibility={gene_reproducibility:.2f}'
                 )
        return
    
    def plot_gene_gradient(self,gene_name):
        """get index and plot a map of a gene name"""
        if self.gene_gradients is None:
            print("First call to gradients, so will take a moment to load...")
            self.gene_gradients = np.load('magicc_expression_data/ahba_vertex_gradients.npy')
        try:
            gene_index = np.where(self.gene_info['gene.symbol']==gene_name)[0][0]
            gene_reproducibility = self.gene_info['Estimated reproducibility'][gene_index]
        except : 
            print(f"{gene_name} not found. Please check spelling or for other aliases")
        msp.plot_surf(self.surf.darrays[0].data,self.surf.darrays[1].data,
                      self.gene_gradients[gene_index],
                  rotate=[90,270],
                 cmap='magma',vmin=0,vmax=0.05,base_size=10,
              mask=~self.cortex_mask,
              mask_colour=np.array([0,0,0,1]),
                  colorbar=True,cmap_label='Expression\ngradient (Z/mm)',
                  title = gene_name+f' expression\ngradient'
                 )
        return
    