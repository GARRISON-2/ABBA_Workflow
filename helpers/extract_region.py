import polars as pl
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from helpers.utils import *




# --- Extract region ids and names from allen reference json ---
def extractAllenInfo(ALLEN_STRUCTS, target_regions):
    with open(ALLEN_STRUCTS, 'r') as file:
        as_json = json.load(file)

    # extract json data and put into flattened list
    all_structures = flatten_structures(as_json['msg'][0])

    # convert to df for easy filtering
    as_df = pl.DataFrame(all_structures)

    # look up IDs from allen LUT
    hid_list = list(set(
        as_df.filter(pl.col('acronym').is_in(target_regions))['id'].to_list()
    ))

    # verify what was found
    found = as_df.filter(pl.col('id').is_in(hid_list))[['id', 'acronym', 'name']]
    print(found)
    print(f"\nTotal regions: {len(hid_list)}")

    return hid_list


def grabROIData(REG_GEO, hid_list, pixel_size_um, sag9):
    # --- Grab corrsponding subregions from the GeoJSON
    with open(REG_GEO, 'r') as file:
        rg_json = json.load(file)


    hippo_data = ROIDATA(polygons=[], ids=[], names=[])
    for feature in rg_json['features']:
        try:
            if feature['properties']['name'] != 'Root':
                cur_id = int(float(feature['properties']['measurements']['ID'])) # some larger IDs in scientific notation for some reason - so must convert to float first

                if cur_id in hid_list:

                    # grab list of all [x,y] coordinate points for the current region
                    cur_coords_raw = feature['geometry']['coordinates'] 
                    
                    # unpack the coordinates into a standard list
                    polygon_list = get_polygons_from_coords(cur_coords_raw)

                    # loop through all the different polygons(individual coordinate sets) within the subregion
                    for i, polygon in enumerate(polygon_list):

                        # convert to um coordinate system the save polygon to coordinate list
                        cur_coords_um = np.array(polygon)[:, :2] * pixel_size_um
                        hippo_data.polygons.append(cur_coords_um)

                        # save the current polygon's subregion id and name
                        hippo_data.ids.append(cur_id)
                        hippo_data.names.append(feature['properties']['name'])


        except (ValueError, KeyError) as e:
            print(f"WARNING - Failed to get Object ID from: {feature['properties']}")
            print(f"  Error: {e}")
            

    if sag9:
        # if ventral hippocampus polygons exist, remove them 
        hippo_data = filterVentral(hippo_data, anchor_ids = {484682470, 502} )
        print(f"Kept {len(hippo_data.polygons)} polygons in correct cluster")

    return hippo_data


# -- Crop out transcript rows for the hippocampus
def cropTranscripts(hippo_data, TR_PATH, CT_OUTPUT):


    transcripts_path = r"S:\Anshutz\Cruz-Martin_Lab\projects\TBI_Project\data\20251013_FT_24hrs_Sag9_ID57476\Transcripts\transcripts.parquet"

    # Lazy loading approach w/ box filter around region of interest:
    scanner = pl.scan_parquet(transcripts_path)

    # grab the length of the parquet from the metadata & return
    full_count = scanner.select(pl.len()).collect().item()
    print(f"Total transcripts in parquet: {full_count:,}")

    # vertically stack all the geoJson coordinate sets into a single np array for easy operations
    stacked_coords = np.vstack(hippo_data.polygons)

    # grab all of the min & max coordinates for x & y
    min_x = stacked_coords[:, 0].min()
    max_x = stacked_coords[:, 0].max()
    min_y = stacked_coords[:, 1].min()
    max_y = stacked_coords[:, 1].max()

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


    print(f"Hippo Box dataframe: {hip_box_df.estimated_size() / 1e9:.2f} GB")


    full_mask = np.zeros(len(hip_box_df), dtype=bool)

    # extract the coordinates into a np array for use in matplot masking
    print("Converting coordinates to np array...")
    t_coords_np = hip_box_df[["x", "y"]].to_numpy()

    for i, region in enumerate(hippo_data.polygons): # for each sub region contained in our full set of hippo coordinates
        print(f"\rChecking Hippocampus polygon: {i + 1}/{len(hippo_data.polygons)}", end="", flush=True)

        # Pre-filter points within this polygon's bounding box
        min_x, min_y = region[:, 0].min(), region[:, 1].min()
        max_x, max_y = region[:, 0].max(), region[:, 1].max()

        bbox_mask = (
            (t_coords_np[:, 0] >= min_x) & (t_coords_np[:, 0] <= max_x) &
            (t_coords_np[:, 1] >= min_y) & (t_coords_np[:, 1] <= max_y)
        )
        bbox_indices = np.where(bbox_mask)[0]

        # create a matplot path object using the current region coordinates
        path = MplPath(region) 

        # create a boolean array of len(hip_box_df), where it is True whenever a coordinate point falls inside the current region
        mask = path.contains_points(t_coords_np[bbox_indices]) 

        # combine masks using an or operation, so that if a coordinate is in any region, it will be saved
        full_mask[bbox_indices[mask]] = True

    # save only the transcripts inside the full hippo mask
    cropped_transcripts = hip_box_df.filter(full_mask)

    print(f"\nCropped transcripts: {len(cropped_transcripts):,} / {full_count:,}")

    # output the cropped df to a csv
    cropped_transcripts.write_csv(CT_OUTPUT, separator="\t")
    print(f"Output cropped transcript tsv to: {CT_OUTPUT}")

    return cropped_transcripts




def makePlot(out_fig, cr_df, target_regions, hippo_data):

    # Top genes
    top_genes = (
        cr_df
        .group_by("gene")
        .agg(pl.col("count").sum())
        .sort("count", descending=True)
        .head(20)
    )


    if len(top_genes) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Top genes
        axes[0].barh(top_genes["gene"], top_genes["count"], color="steelblue")
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Transcript count")
        axes[0].set_title("Top 20 genes in ROI")
        
        # Plot 2: Spatial distribution (sample)
        sample_cropped = cr_df.sample(n=min(10000, len(cr_df)))
        axes[1].scatter(sample_cropped["x"], sample_cropped["y"], s=0.1, alpha=0.3, c='blue')
        #axes[1].plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2)
        axes[1].set_aspect('equal')
        axes[1].set_title("Sampled Transcripts in ROI")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        
        # Color polygons by region
        region_colors = plt.cm.tab20(np.linspace(0, 1, len(target_regions)))
        region_color_map = dict(zip(target_regions, region_colors))

        # loop through all of the features in the geojson
        for i, polygon in enumerate(hippo_data.polygons):
    
            # plot the current polygon outline 
            axes[1].plot(polygon[:, 0], polygon[:, 1], '-',
                linewidth=1.5,
                color=region_color_map.get(hippo_data.names[i], 'red'),
                label=hippo_data.names[i])

        plt.tight_layout()
        plt.savefig(out_fig)
        print(f"Saved cropping figure to: {out_fig}")