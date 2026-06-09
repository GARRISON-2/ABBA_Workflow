import numpy as np
from scipy.cluster.vq import kmeans2
from dataclasses import dataclass

@dataclass
class ROIDATA:
    polygons: list
    ids: list # list of all UNIQUE region ids
    names: list # list of region names that correspond with each polygon

# recursively extract all structures
def flatten_structures(node, parent_name=None):
    structures = [{
        'id': node['id'],
        'name': node['name'],
        'acronym': node['acronym'],
        'parent_structure_id': node.get('parent_structure_id'),
        'parent_name': parent_name,
        'graph_order': node.get('graph_order'),
        'st_level': node.get('st_level'),
        'depth': node.get('depth')
    }]
    
    for child in node.get('children', []):
        structures.extend(flatten_structures(child, parent_name=node['name']))
    
    return structures


def get_polygons_from_coords(coords):
    if isinstance(coords[0][0][0], (int, float)):
        # coordinates are not nested, so can be unpacked normally
        return [coords[0]]
    else:
        # coordinates are nested and need to be looped through
        return [poly[0] for poly in coords]


def filterVentral(hp_data, anchor_ids):
    # cluster by region polygon centroid
    centroids = np.array([p.mean(axis=0) for p in hp_data.polygons])
    _, labels = kmeans2(centroids, 2, seed=42)

    # find which cluster contains anchors (ie cluster 0 or cluster 1, etc.)
    correct_cluster = None
    for label_type in set(labels):     # loop through label cluster

        # get all subregion IDs that are within the current cluster
        cluster_ids = {hp_data.ids[index] for index, label in enumerate(labels) if label == label_type}
        
        if anchor_ids & cluster_ids:  # if cluster contains ProS or SUB
            correct_cluster = label_type
            break

    # loop and grab indices of all correct polygons
    indices = [i for i, label in enumerate(labels) if label == correct_cluster]

    # grab all correct polygons and their associated region names
    hp_data.polygons = [hp_data.polygons[i] for i in indices]
    hp_data.names = [hp_data.names[i] for i in indices]

    return hp_data
