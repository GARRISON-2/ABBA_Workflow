import polars as pl
import json
from pathlib import Path
import os
import numpy as np
from matplotlib.path import Path as MplPath

import gzip

# Recursively extract all structures
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


# Recursively find all children of specific parent id
def find_children(descendants, parent_id, df):
    # grab all columns with specific parent id
    children = df.filter(
        pl.col('parent_structure_id') == parent_id
    )['id'].to_list()

    # loop through all children with chosen parent id
    for child_id in children:
        # add current child id to full descendant list
        descendants.append(child_id)

        # check if current child has any of its own children
        find_children(descendants, child_id, df)

    return descendants


# Filter for region and descendants
def GetRegionAll(df, region_name):
    """Get a region and all its subregions"""
    # Find the parent region
    parent = df.filter(
        pl.col('name').str.contains("(?i)" + region_name) # (?i) adds case insensitivity
    )

    # if the name was not found:
    # if parent.empty:
    #     return pl.DataFrame()
    
    parent_id = parent["id"][0]
    
    # Get all descendants
    descendants = find_children([parent_id], parent_id, df)
    
    # return the filtered dataframe, as well as a list of the IDs
    return df.filter(pl.col('id').is_in(descendants)), descendants


# def get_polygons_from_coords(coords):
#     """
#     Dynamically handle coordinate nesting regardless of geometry type.
#     Returns a list of coordinate arrays.
#     """
#     # Check nesting depth of first element
#     first = coords[0]
    
#     if isinstance(first[0], (int, float)):
#         # [x, y] - single coordinate, shouldn't happen at this level
#         return [coords]
#     elif isinstance(first[0][0], (int, float)):
#         # [[x,y], [x,y], ...] - already a ring
#         return [first][0]
#     elif isinstance(first[0][0][0], (int, float)):
#         # [[[x,y], ...]] - MultiPolygon style
#         return [p[0][0] for p in coords]
#     else:
#         # Even deeper nesting
#         return [p[0][0][0] for p in coords]

def get_polygons_from_coords(coords):
    print(f"coords type: {type(coords)}")
    print(f"coords length: {len(coords)}")
    print(f"coords[0] type: {type(coords[0])}")
    print(f"coords[0][0] type: {type(coords[0][0])}")
    print(f"coords[0][0][0] type: {type(coords[0][0][0])}")
    try:
        print(f"coords[0][0][0][0] type: {type(coords[0][0][0][0])}")
    except:
        print("coords[0][0][0][0]: does not exist")
    print("---")
    return []

# def GetTransformInfo(qp_file):
#     with open(qp_file, 'rt', encoding='utf-8') as f:
#         qp_data = json.load(f)

#     # Look for transformation matrix
#     print("=== Top-level keys ===")
#     print(qp_data.keys())


PROJ_ROOT = Path(__file__).resolve().parents[1]
ALLEN_STRUCTS = PROJ_ROOT / "data" / "allen_structures.json"

REG_GEO = r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\QP_test_2\aligned_0.geojson"
# qp_data = r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\QuPath_Project\data\3\data.qpdata"

pixel_size_um = 0.2125


# --- Extract region ids and names from allen reference json ---

with open(ALLEN_STRUCTS, 'r') as file:
    as_json = json.load(file)

all_structures = flatten_structures(as_json['msg'][0])
as_df = pl.DataFrame(all_structures)

# Save the full LUT
# as_df.write_csv('all_regions.csv')

print(as_df["name"].head())

# Get list of ids for hippocampus and all subregions
hr_df, hid_list = GetRegionAll(as_df, "Hippocampal formation")

# output csv containing all hippocampus regions that will be looked for
hr_df.write_csv('hip_regions.csv')

print("Hippocampus regions found:")
print(hr_df[['id', 'acronym', 'name']])
print(f"\nTotal regions: {len(hr_df)}")
print(f"\nHippocampus region IDs: {hr_df['id'].to_list()}")


# --- Grab corrsponding regions from the GeoJSON

with open(REG_GEO, 'r') as file:
    rg_json = json.load(file)


#GetTransformInfo(qp_data)


hippo_coords = []
for feature in rg_json['features']:
    try:
        if feature['properties']['name'] != 'Root':
            cur_id = int(float(feature['properties']['measurements']['ID'])) # some larger IDs in scientific notation for some reason - so must convert to float first

            if cur_id in hid_list:
                cur_coords_raw = feature['geometry']['coordinates'] # grab list of all [x,y] coordinate points for the current region
                print(cur_id)
                # unpack the coordinates into a standard list
                cur_coords = get_polygons_from_coords(cur_coords_raw)

                # translate the ABBA coordinates into the same coordinate space as the Xenium transcsripts use:
                #cur_coords_um = [[coord[0] * pixel_size_um, coord[1] * pixel_size_um] for coord in cur_coords]

                # cur_coords_um = []
                # for coord in cur_coords:

                #     cur_coords_um. append([coord[0] * pixel_size_um, coord[1] * pixel_size_um])

                # hippo_coords.append(np.array(cur_coords_um))

    except (ValueError, KeyError) as e:
        print(f"WARNING - Failed to get Object ID from: {feature['properties']}")
        print(f"  Error: {e}")
        
print(f"Extracted {len(hippo_coords)} relevant hippocampus coordinate regions")


# -- crop out the transcript rows for the hippocampus

#transcripts_path = r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\data\Yaseer_Example\InputTranscripts.tsv"
transcripts_path = r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\data\20251013_FT_24hrs_Sag9_ID57476\Transcripts\transcripts.parquet"

# Lazy loading approach w/ box filter around region of interest:
scanner = pl.scan_parquet(transcripts_path)

# grab the length of the parquet from the metadata & return
full_count = scanner.select(pl.len()).collect().item()
print(f"Total transcripts in parquet: {full_count:,}")

# vertically stack all the geoJson coordinate sets into a single np array for easy operations
stacked_coords = np.vstack([region[0] for region in hippo_coords])

# grab all of the min & max coordinates for x & y
min_x = stacked_coords[:, 0].min()
max_x = stacked_coords[:, 0].max()
min_y = stacked_coords[:, 1].min()
max_y = stacked_coords[:, 1].max()

print("=== GeoJSON Bounds (ABBA space) ===")
print(f"X: {min_x:.1f} to {max_x:.1f}")
print(f"Y: {min_y:.1f} to {max_y:.1f}")
print(f"Width: {max_x - min_x:.1f}")
print(f"Height: {max_y - min_y:.1f}")

# load transcripts to df that are inside hippocampus box and format for punst tsv
print("Attempting to load transcript parquet file...")
hip_box_df = scanner.select(
    # select these columns with simplified column names:
    pl.col("x_location").alias("x").cast(pl.Float32),
    pl.col("y_location").alias("y").cast(pl.Float32),
    pl.col("feature_name").alias("gene").cast(pl.Categorical),
    pl.lit(1).cast(pl.Int16).alias("count") # set the count column to 1 on every row
).filter(
    # only grab rows that are within a box formed by the min and max hippo coordinates
    (pl.col("x") >= min_x) & (pl.col("x") <= max_x) &
    (pl.col("y") >= min_y) & (pl.col("y") <= max_y)
).collect() # load only the regions within our box into the dataframe


# DEBUG - reading in whole parquet
# try:
#     print("Attempting to load transcript parquet file...")
#     hip_box_df = (
#         pl.read_parquet(transcripts_path)
#         .select([
#             pl.col("x_location").alias("x").cast(pl.Float32), # read in parquet 
#             pl.col("y_location").alias("y").cast(pl.Float32),
#             pl.col("feature_name").alias("gene").cast(pl.Categorical)
#         ])
#     )
# except FileNotFoundError:
#     print(f"Error: File not found at {transcripts_path}")
#     exit()
# print(f"Total transcripts: {len(hip_box_df):,}")


print(f"Hippo Box dataframe: {hip_box_df.estimated_size() / 1e9:.2f} GB")

# DEBUG:
print(hip_box_df.head())


# create empty mask array to hold
full_mask = np.zeros(len(hip_box_df), dtype=bool)

# extract the coordinates into a np array for use in matplot masking
print("Converting coordinates to np array...")
t_coords_np = hip_box_df[["x", "y"]].to_numpy()

trans_min_x = t_coords_np[:, 0].min()
trans_max_x = t_coords_np[:, 0].max()
trans_min_y = t_coords_np[:, 1].min()
trans_max_y = t_coords_np[:, 1].max()

# DEBUG
print("=== Transcript Bounds (Xenium Space) ===")
print(f"X: {trans_min_x:.1f} to {trans_max_x:.1f}")
print(f"Y: {trans_min_y:.1f} to {trans_max_y:.1f}")
print(f"Width: {trans_max_x - trans_min_x:.1f}")
print(f"Height: {trans_max_y - trans_min_y:.1f}")
print(f"Coords array: {t_coords_np.nbytes / 1e9:.2f} GB")


for i, region in enumerate(hippo_coords): # for each sub region contained in our full set of hippo coordinates
    print(f"\rChecking Hippocampus region: {i}/{len(hippo_coords)}", end="", flush=True)

    # create a matplot path object using the current region coordinates
    path = MplPath(region) 

    # create a boolean array of len(hip_box_df), where it is True whenever a coordinate point falls inside the current region
    mask = path.contains_points(t_coords_np) 

     # combine masks using an or operation, so that if a coordinate is in any region, it will be saved
    full_mask |= mask

# save only the transcripts inside the full hippo mask
cropped_transcripts = hip_box_df.filter(full_mask)

# output the cropped df to a csv
cropped_transcripts.write_csv("cropped_transcripts.tsv", separator="\t")

print(f"Cropped transcripts: {len(cropped_transcripts):,} / {full_count:,}")


sample_cropped = cropped_transcripts.sample(n=min(10000, len(cropped_transcripts)), random_state=42)
axes[1].scatter(sample_cropped["x"], sample_cropped["y"], s=0.1, alpha=0.3, c='blue')
axes[1].plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2)
axes[1].set_aspect('equal')
axes[1].set_title("Transcripts in ROI")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

plt.show()

