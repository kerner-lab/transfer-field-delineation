import os
import subprocess
import tensorflow as tf
from pprint import pprint
import scipy.misc
import numpy as np
import sys
import csv 
import imageio
options = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
root = "data/"

filenames = ["./data/sentinel_tf/" + f for f in os.listdir("./data/sentinel_tf/")]
print(filenames)
idx = 0
for f in filenames:
    f_sub = f.split('.')[1][10:]
    print(f_sub)
    output_directory = root + "/sentinel/" + f_sub
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    satellite_features = ['B1', 'B2', 'B3']

    print(">>>>>> Processing: " + f)
    iterator = tf.compat.v1.python_io.tf_record_iterator(f, options=options)
    n = 0
    print(f)
    iter = 10
    csv_file = output_directory  + "/img_csv.csv"
    csv_file_keys = ['Parcel_id', 'max_lat', 'max_lon', 'min_lat', 'min_lon']
    with open(csv_file, 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(csv_file_keys)
    
    while iter > 0:
        with open(csv_file, 'a') as f_csv:
            csv_writer = csv.writer(f_csv)
            try:
                record_str = next(iterator)
                ex = tf.train.Example.FromString(record_str)
                #print(ex.features)
                min_lon = min(ex.features.feature['longitude'].float_list.value) 
                max_lon = max(ex.features.feature['longitude'].float_list.value) 
                min_lat = min(ex.features.feature['latitude'].float_list.value) 
                max_lat = max(ex.features.feature['latitude'].float_list.value) 
                idx = idx + 1#int(ex.features.feature['Parcel_id'].float_list.value[0])
                features = []
                for satellite_feature in satellite_features:
                    feature = (ex.features.feature[satellite_feature].float_list.value)
                    feature = np.array(feature)
                    feature = feature.reshape((225, 225, 1))
                    feature = np.flip(feature, axis=0)
                    features.append(feature)

                csv_writer.writerow([idx, max_lat, max_lon, min_lat, min_lon])
                image = np.concatenate(features, axis=2)
                image = image[:224, :224, :]

                if idx != -1:
                    jpeg_path = output_directory + '/' + str(idx) + '.jpeg'
                    
                    imageio.imwrite(jpeg_path, image)
          
                    #scipy.misc.toimage(image, cmin=0.0, cmax=...).save(jpeg_path)
                    #writer = tf.python_io.TFRecordWriter(tfrecord_path, options=options)
                    #writer.write(ex.SerializeToString())
                    #writer.close()
                #print(idx)
                n += 1
                if n%10==0:
                    print("       Processed " + str(n) + " records in " + f)
            except Exception as e:
                iter -= 1
                print(e)
                print(">>>>>> Processed " + str(n) + " records in " + f)

