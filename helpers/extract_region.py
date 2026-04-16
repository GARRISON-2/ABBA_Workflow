import polars as pl
import json
from pathlib import Path
import os


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
def find_children(descendants, parent_id):
    # grab all columns with specific parent id
    children = df.filter(
        pl.col('parent_structure_id') == parent_id
    )['id'].to_list()

    # loop through all children with chosen parent id
    for child_id in children:
        # add current child id to full descendant list
        descendants.append(child_id)

        # check if current child has any of its own children
        find_children(descendants, child_id)

    return descendants


# Filter for region and descendants
def get_region_and_descendants(df, region_name):
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
    descendants = find_children([parent_id], parent_id)
    
    return df.filter(pl.col('id').is_in(descendants))


PROJ_ROOT = Path(__file__).resolve().parents[1]
ALLEN_STRUCTS = PROJ_ROOT / "data" / "allen_structures.json"

REG_GEO = ""

# --- Extract region ids and names from allen reference json ---

with open(ALLEN_STRUCTS, 'r') as file:
    as_json = json.load(file)

all_structures = flatten_structures(as_json['msg'][0])
df = pl.DataFrame(all_structures)

# Save the full LUT
# df.write_csv('allen_brain_lut.csv', index=False)

print(df["name"].head())
# Get hippocampus and all subregions
hippo_regions = get_region_and_descendants(df, 'Hippocampal formation')
hippo_regions.write_csv('regions.csv')


print("Hippocampus regions found:")
print(hippo_regions[['id', 'acronym', 'name']])
print(f"\nTotal regions: {len(hippo_regions)}")
print(f"\nHippocampus region IDs: {hippo_regions['id'].to_list()}")


# ---grab all coordinate sets that match with the region IDs---

with open(REG_GEO, 'r') as file:
    as_json = json.load(file)




