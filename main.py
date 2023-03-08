import osmapi
import pandas as pd
import osmnx as ox
import geopandas as gpd







def print_hi(name):
    north, south, east, west = 40.7527, 40.7309, -73.9776, -74.0060
    G = ox.graph_from_bbox(north, south, east, west, network_type='all')#, tags={'building': True})
    # extract the buildings within the bounding box
    buildings = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})

    # extract the building types from the buildings
    building_types = list(set(buildings['building'].values))
    num_buildings = len(buildings)
    print(f"There are {num_buildings} buildings in {place_name}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


