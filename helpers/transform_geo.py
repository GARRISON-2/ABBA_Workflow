import numpy as np
import json


# Function to transform GeoJSON coordinates
def transformGeojsonCoords(geojson, transform_matrix):
    """Apply transformation to all polygon coordinates"""
    for feature in geojson['features']:
        coords = np.array(feature['geometry']['coordinates'][0])
        
        # Add z=0 and homogeneous coordinate
        n_points = len(coords)
        coords_3d = np.hstack([coords, np.zeros((n_points, 1)), np.ones((n_points, 1))])
        
        # Apply transformation
        transformed = (transform_matrix @ coords_3d.T).T
        
        # Extract x, y (drop z and homogeneous coordinate)
        coords_2d = transformed[:, :2]
        
        feature['geometry']['coordinates'][0] = coords_2d.tolist()
    
    return geojson


def parse_affine_3d(affine_list):
    """Convert ABBA's flattened 3D affine to 4x4 matrix"""
    # ABBA stores as [m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23]
    # We need a 4x4 homogeneous matrix
    matrix = np.array([
        [affine_list[0], affine_list[1], affine_list[2], affine_list[3]],
        [affine_list[4], affine_list[5], affine_list[6], affine_list[7]],
        [affine_list[8], affine_list[9], affine_list[10], affine_list[11]],
        [0, 0, 0, 1]
    ])
    return matrix


def transform(transform_json, out_json_name):

    # Load the transformation file
    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    # Parse all transformations
    transform_matrices = []
    for i in range(transforms['size']):
        key = f'realTransform_{i}'

        try:
            affine_list = transforms[key]['affinetransform3d']
        except:
            affine_list = transforms[key]['realTransform']


        matrix = parse_affine_3d(affine_list)
        transform_matrices.append(matrix)
        print(f"\nTransform {i}:")
        print(matrix)

    # Combine: forward transform = T0 * T1 * T2 * T3 * T4
    forward = np.eye(4)
    for T in transform_matrices:
        forward = forward @ T

    print("\n=== Combined Forward Transform ===")
    print(forward)

    # Invert to go from atlas → xenium
    inverse = np.linalg.inv(forward)

    print("\n=== Inverse Transform (Atlas → Xenium) ===")
    print(inverse)

    # Load and transform your GeoJSON
    with open('atlas.geojson', 'r') as f:
        geojson = json.load(f)

    transformed_geojson = transformGeojsonCoords(geojson, inverse)

    # Save transformed GeoJSON (now in Xenium micron space)
    with open(out_json_name, 'w') as f:
        json.dump(transformed_geojson, f, indent=2)

    print("\nTransformed GeoJSON saved to atlas_xenium_space.geojson")




transform(r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\QP_test_2\data\1\ABBA-Transform-allen_mouse_10um_java.json", r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\ABBA_Workflow\transformed_ABBA_registered.geojson")

