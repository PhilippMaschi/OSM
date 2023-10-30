from osgeo import gdal, osr
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.windows import from_bounds
from pathlib import Path
import pyproj
from shapely.ops import transform
from shapely.geometry import box, shape
from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA, BASE_EPSG


def transform_bounds(east, west, south, north, input_epsg, output_epsg):
    # Define the source and destination SRS
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(input_epsg)

    dest_srs = osr.SpatialReference()
    dest_srs.ImportFromEPSG(output_epsg)

    # Set up the transformation
    transform = osr.CoordinateTransformation(source_srs, dest_srs)

    # Transform the bounds
    east_trans, north_trans, _ = transform.TransformPoint(east, north)
    west_trans, south_trans, _ = transform.TransformPoint(west, south)

    return east_trans, west_trans, south_trans, north_trans


def extract_subset_from_tif(city: dict):
    """
    File from: https://doi.org/10.2909/42690e05-edf4-43fc-8020-33e130f62023
    :param city: dict from main
    :return:
    """
    input_file = r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\copernicus_r_3035_10_m_ua-bh-2012_p_2012_v03_r00\GTiff\Building_Height_UA2012_WM_10m.tif"
    ds = gdal.Open(input_file)
    # Convert geographic coordinates to pixel coordinates
    geotransform = ds.GetGeoTransform()
    # find out the epsg of the tiff file:
    wkt_projection = ds.GetProjection()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(wkt_projection)
    epsg_code = spatial_ref.GetAttrValue("AUTHORITY", 1)
    # transform the coordinates from main to the coordinates used in the tiff file:
    east_trans, west_trans, south_trans, north_trans = transform_bounds(
        city["east"], city["west"],  city["south"], city["north"], BASE_EPSG, int(epsg_code)
    )

    def geo_to_pixel(geo_x, geo_y, geotransform):
        """
        Convert geographic coordinates to pixel coordinates.
        """
        pixel_x = int((geo_x - geotransform[0]) / geotransform[1])
        pixel_y = int((geo_y - geotransform[3]) / geotransform[5])
        return pixel_x, pixel_y

    west_pixel, north_pixel = geo_to_pixel(west_trans, north_trans, geotransform)
    east_pixel, south_pixel = geo_to_pixel(east_trans, south_trans, geotransform)
    assert north_pixel <= south_pixel, "North and South pixel values are inverted."
    assert west_pixel <= east_pixel, "West and East pixel values are inverted."

    # Compute pixel dimensions of the subset
    x_offset = west_pixel
    y_offset = north_pixel
    x_size = east_pixel - west_pixel
    y_size = south_pixel - north_pixel
    # Ensure the subset is within the raster's boundaries
    if (x_offset < 0 or y_offset < 0 or
            (x_offset + x_size) > ds.RasterXSize or
            (y_offset + y_size) > ds.RasterYSize):
        raise ValueError("Requested subset is outside the raster boundaries.")

    # Read the subset of data
    subset = ds.GetRasterBand(1).ReadAsArray(x_offset, y_offset, x_size, y_size)

    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    output_file = r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\copernicus_r_3035_10_m_ua-bh-2012_p_2012_v03_r00" + "\copernicus_{city}.tif"
    out_ds = driver.Create(output_file, x_size, y_size, ds.RasterCount, gdal.GDT_Float32)
    out_ds.SetGeoTransform((ulx, geotransform[1], 0, uly, 0, geotransform[5]))

    # Write the subset to the output file
    for i in range(ds.RasterCount):
        out_ds.GetRasterBand(i + 1).WriteArray(subset)

    out_ds = None
    ds = None


if __name__ == "__main__":
    # Example usage
    # Geotransform coordinates are in the order: [upper left x, lower right y, lower right x, upper left y]
    extract_subset_from_tif(MURCIA)










