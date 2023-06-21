import time
import fiona
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


def calc_premeter(input_lyr, output_lyr, positive_buffer, negative_buffer):
    st = time.time()
    if positive_buffer + negative_buffer >= 0:
        print("Absolut value of the negative buffer should be greater than the positive buffer")
        return
    orig_poly = [shape(feature['geometry']) for feature in fiona.open(input_lyr)]
    # +positive_buffer meter buffer
    orig_poly_buffer = [shape(feature['geometry']).buffer(positive_buffer) for feature in fiona.open(input_lyr)]
    # polygon union
    poly_buffer_union = unary_union(orig_poly_buffer)
    # negative_buffer meter buffer: overlays inside the original layer
    boundary_buffer_union = [g.buffer(negative_buffer).boundary for g in poly_buffer_union]
    buffer_union = unary_union(boundary_buffer_union)
    premeter = []
    # use Sort-Tile-Recursive for spatial query
    tree = STRtree(buffer_union)
    for i, item in enumerate(orig_poly):
        buffer_union_query = unary_union(tree.query(item))
        temp_segment = item.intersection(buffer_union_query)
        accumulator = 0
        if temp_segment.geom_type == "MultiLineString":
            accumulator += sum([g.length for g in temp_segment])
        elif temp_segment.geom_type == "LineString":
            accumulator = temp_segment.length
        else:
            print('Check item: ', i)
        premeter.append(accumulator)

    gdf = gpd.read_file(input_lyr)
    gdf['exPremeter'] = np.array(premeter)
    gdf.to_file(output_lyr)
    gdf = None
    # calculate deviation
    approx_extern_premeter = sum([elem.length for elem in buffer_union])
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


if __name__ == "__main__":
    path = 'C:/Users/Mostafa/Desktop/QGIS_Shapes_ETRS/'
    input_lyr = path + 'selection_3035.shp'
    output_lyr = path + 'outside_premeter.shp'
    positive_buffer = 0.1
    negative_buffer = -0.11
    calc_premeter(input_lyr, output_lyr, positive_buffer, negative_buffer)
