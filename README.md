## Parcel Delineation

This is a modified Python code repository for parcel delineation using satellite imagery. It contains scripts for training and evaluating U-Net models, as well as for preparing the dataset.

To train a model, you can use the train_unet.py script. This script has a number of parameters that you can tweak, such as whether to use a dilated U-Net, a pretrained U-Net, or a stacked U-Net. By default, the model uses a pretrained stacked U-Net.

To evaluate a model, you can use the predict_model.py script. This script takes as input the path to the saved model and the path to a CSV file that contains the paths to the images and masks that you want to evaluate. The script will output the prediction numpy array of the model, as well as the F1 score and accuracy.

To prepare the dataset, you can use the following scripts:

utils/sample_shp.py: This script samples random polygons from the French polygons dataset.

utils/get_centroid.py: This script reads the shape file to get the centroid of each polygon.

utils/convert_tfrecords_jpeg.py: This script extracts jpegs from the tfrecord format satellite images and also creates a CSV file that contains the max lat, max lon, min lat, and min lon of each image.

utils/shp2geo.py: This script gets only polygons that overlap in bounds of extracted images.

utils/create_mask.py: This script creates the masks (boundary and filled) of the extracted polygons and images.

utils/split_data.py: This script splits the data into train/test/val sets.
