
import numpy as np

def get_boundary(roi1,roi2,neighbours,parcellation):
    """get boundary vertices between to rois"""
    labels_dict = parcellation.labeltable.get_labels_as_dict()
    inv_reg_dict = {v: k for k, v in labels_dict.items()}
    parcellation = parcellation.darrays[0].data
    region_index1 = inv_reg_dict[roi1]
    region_index2 = inv_reg_dict[roi2]
    r1 = neighbours[parcellation==region_index1]
    r2 = neighbours[parcellation==region_index2]
    mv = []
    for v in r1:
        mv.extend(v)
    sv = []
    for v in r2:
        sv.extend(v)
    mv = np.unique(mv)
    sv = np.unique(sv)
    border =np.intersect1d(sv,mv)
    return border




def get_neighbours_from_tris(tris, label=None):
    """Get surface neighbours from tris
        Input: tris
         Returns Nested list. Each list corresponds 
        to the ordered neighbours for the given vertex"""
    n_vert=np.max(tris+1)
    neighbours=[[] for i in range(n_vert)]
    for tri in tris:
        neighbours[tri[0]].extend([tri[1],tri[2]])
        neighbours[tri[2]].extend([tri[0],tri[1]])
        neighbours[tri[1]].extend([tri[2],tri[0]])
    #Get unique neighbours
    for k in range(len(neighbours)):      
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else :
            neighbours[k]=f7(neighbours[k])
    return np.array(neighbours,dtype=object)



def f7(seq):
    #returns uniques but in order to retain neighbour triangle relationship
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]