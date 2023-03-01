from mpl3d import glm
from mpl3d.mesh import Mesh
import numpy as np



class Surface(Mesh):
    def __init__(self, ax, transform, Z, border=False,vertices=None,faces=None,
                 facecolors=None, *args, **kwargs):
        if border:
            n2, n1 = Z.shape
            Z_ = np.zeros((n2+2, n1+2))
            Z_[1:-1,1:-1] = Z
            Z = Z_
            x = np.zeros(n1+2)
            x[1:-1] = np.linspace(-0.5, +0.5, n1)
            x[0], x[-1] = x[1], x[-2]
            y = np.zeros(n2+2)
            y[1:-1] = np.linspace(-0.5, +0.5, n2)
            y[0], y[-1]  = y[1], y[-2]
            F = kwargs["facecolors"]
            F_ = np.zeros((F.shape[0]+2,F.shape[1]+2,F.shape[2]))
            F_[1:-1,1:-1] = F
            F_[0,:] = F_[1,:]
            F_[-1,:] = F_[-2,:]
            F_[:,0] = F_[:,1]
            F_[:,-1] = F_[:,-2]
            kwargs["facecolors"] = F_
        # Recompute colors for triangles based on vertices color
        facecolors =  np.median(facecolors[faces],axis=-2)
        facecolors = facecolors.reshape(-1,4)[:,:3]
        F = vertices[faces]
        # Light direction
        direction = glm.normalize([1,1,1])
        # Faces center
        C = F.mean(axis=1)
        # Faces normal
        N = glm.normalize(np.cross(F[:,2]-F[:,0], F[:,1]-F[:,0]))
        # Relative light direction
        D = glm.normalize(C - direction)
        # Diffuse term
        diffuse = glm.clip((N*D).sum(-1).reshape(-1,1))
        facecolors = (1-diffuse)*facecolors
        kwargs["facecolors"] = facecolors
        Mesh.__init__(self, ax, transform, vertices, faces,
                       *args, **kwargs)