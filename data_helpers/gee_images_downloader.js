/**
 * @fileoverview Downloads the S2/Planet Images given the coordinates of the region of interest
 * @param {ee.Geometry} region The region of interest
 * @param {string} output_path The path where the images will be downloaded
 * @param {string} start_date The start date of the images to download
 * 
 */


var region =  ee.Geometry.Polygon([
    [[-0.7155534667066932, 43.39057963239615],
      [-0.7155534667066932, 43.156608691390645],
      [0.34359722909410806, 43.156608691390645],
      [0.34359722909410806, 43.39057963239615]]
]);


var start_date = '2021-01-01';
var end_date = '2021-12-31';

var ic = ee.ImageCollection('your_image_collection_here')
// var ic = ee.ImageCollection('COPERNICUS/S2_SR')


Map.addLayer(region);

Map.centerObject(region, 9);

function addLatLong(image) {
    var latlon = ee.Image.pixelLonLat()//.reproject(proj)
    return image.addBands(latlon.select('longitude','latitude'))
}

// Get the image from the collection
var region_images = ic.filterBounds(region)
                    .filter(ee.Filter.lt('cloud_cover', 0.2))
                    .map(addLatLong)
                    .map(function(image){return image.clip(region)})
// var regiion_images = ic.filterBounds(region)
//                         .filterDate(start_date, end_date)
//                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
//                         .map(function(image){return image.clip(region)})

region_images = region_images.median();


var rgbVis = {
    min: 180,
    max: 2000,
    bands: ['B3', 'B2', 'B1'],
};
// var rgbVis = {
//     min: 180,
//     max: 2000,
//     bands: ['B4', 'B3', 'B2'],
// };


Map.addLayer(
    region_images,
    rgbVis,
    '2021-France', true
);

var image_export_options = {
    'patchDimensions': [224, 224],
    'maxFileSize': 104857600,
    'compressed': true
}


Export.image.toDrive({
  image: scene.select(['B1', 'B2', 'B3','longitude','latitude']),
  description: 'Planet-France-21',
  fileNamePrefix: 'ps_france_21',
  scale: 3,
  folder: 'ps_france_21',
  fileFormat: 'TFRecord',
  region: region,
  formatOptions: image_export_options,  
  maxPixels: 100000000000
});
// Export.image.toDrive({
//     image: scene.select(['B4', 'B3', 'B2','longitude','latitude']),
//     description: 'S2-France-21',
//     fileNamePrefix: 's2_france_21',
//     scale: 10,
//     folder: 's2_france_21',
//     fileFormat: 'TFRecord',
//     region: region,
//     formatOptions: image_export_options,  
//     maxPixels: 100000000000
//   });