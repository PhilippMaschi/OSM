import time
import fiona
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from pathlib import Path


def calc_premeter(input_lyr: Path,
                  output_lyr: Path,
                  positive_buffer: float = 0.1,
                  negative_buffer: float = -0.11):
    """
    Achtung, die Koordinaten müssen für diese Funktion metrisch sein!
    :param input_lyr: path to .shp file
    :param output_lyr: path to where output file should be saved (.shp)
    :param positive_buffer: used to calculate the areas where buildings touch
    :param negative_buffer: absolute value should be greater than positive buffer
    :return: geopandas dataframe from the input layer with two additional rows describing the free (not adjacent) length
                and the circumference (both in m)

    """
    # checking if csr is correct
    # gdf = gpd.read_file(input_lyr)
    # if gdf.crs.to_epsg() != 3035:
    #     gdf.to_crs("epsg:3035")
    #     gdf.to_file(input_lyr, driver="GPKG")

    print(f"calculating free length of each building and circumference...")
    st = time.time()
    if positive_buffer + negative_buffer >= 0:
        print("Absolut value of the negative buffer should be greater than the positive buffer")
        return
    orig_poly = []
    for feature in fiona.open(input_lyr):
        geometry = shape(feature['geometry'])
        if geometry.geom_type == "Polygon" and geometry.is_valid:
            orig_poly.append(geometry)

    # +positive_buffer meter buffer
    orig_poly_buffer = [geometry.buffer(positive_buffer) for geometry in orig_poly]
    # polygon union
    poly_buffer_union = unary_union(orig_poly_buffer)
    # negative_buffer meter buffer: overlays inside the original layer
    boundary_buffer_union = [g.buffer(negative_buffer).boundary for g in poly_buffer_union.geoms]

    # Combine the boundary buffers
    buffer_union = unary_union(boundary_buffer_union)
    # Now, calculate the perimeter of the union of buffers (positive and negative combined)
    premeter = []
    # also calculate the total curcumference of each object:
    circumference = []
    # use Sort-Tile-Recursive for spatial query
    tree = STRtree(buffer_union.geoms)

    for i, item in enumerate(orig_poly):
        # query from the tree returns the indices from buffer_union
        tree_query = tree.query(item)
        # Get the actual intersecting geometries using the indices
        intersecting_geometries = [buffer_union.geoms[index] for index in tree_query]
        # create unary union from the intersecting geometries
        buffer_union_query = unary_union(intersecting_geometries)
        # calculate circumference:
        # if buffer_union_query.geom_type == "LineString":
        circumference.append(item.length)
        # elif buffer_union_query.geom_type == "MultiLineString":
        #     circumference.append([number for number in [l.length for l in buffer_union_query.geoms]])
        # else:
        #     print(f"check item number {i}")
        # create temporary segment of the intersections to save time
        temp_segment = item.intersection(buffer_union_query)
        accumulator = 0
        if temp_segment.geom_type == "MultiLineString":
            accumulator += sum([g.length for g in temp_segment.geoms])
        elif temp_segment.geom_type == "LineString":
            accumulator = temp_segment.length
        else:
            print('Check item: ', i)
        premeter.append(accumulator)


    gdf = gpd.read_file(input_lyr)
    gdf['free length (m)'] = np.array(premeter)
    gdf['circumference (m)'] = np.array(circumference)
    gdf.to_file(output_lyr)

    # calculate deviation
    approx_extern_premeter = sum([elem.length for elem in buffer_union.geoms])
    returned_premeter = sum(premeter)
    approx_dev_percent = 100 * (approx_extern_premeter - returned_premeter) / approx_extern_premeter
    elapsed = time.time() - st
    '''
    # -positive_buffer meter buffer: overlays on the original layer
    boundary_buffer_union_org = [g.buffer(-positive_buffer).boundary for g in poly_buffer_union]
    expected_extern_premeter = sum([elem.length for elem in unary_union(boundary_buffer_union_org)])
    dev_percent = 100*(expected_extern_premeter-returned_premeter)/expected_extern_premeter
    print("Deviation from expected premeter: %0.2f%% " %dev_percent)
    '''
    print("Deviation from approximate premeter: %0.2f%% " % approx_dev_percent)
    print("Elapsed time: %0.2f" % elapsed)
    print("free length and circumference added to gdf")
    return gdf

if __name__ == "__main__":
    path = Path(r'C:\Users\mascherbauer\PycharmProjects\OSM')
    input_lyr = path / 'merged_osm_geom.shp'
    output_lyr = path / 'new_merged_osm_geom.shp'
    positive_buffer = 0.1
    negative_buffer = -0.11
    calc_premeter(input_lyr, output_lyr, positive_buffer, negative_buffer)
