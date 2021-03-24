import argparse
import os
import glob
import boto3
import tensorflow as tf
import time

def get_s3_bucket(bucket_name):
    """Return S3 bucket connection.
    Args:
        bucket_name: name of the bucket
    Return:
        S3 bucket connection
    """

    s3 = boto3.resource('s3')
    return s3.Bucket(bucket_name)


def download_bucket(bucket, save_path):
    """Download buckets content into save_path"""
    print("Started downloading {} bucket into {} ...".format(bucket, save_path))
    curr_time = time.time()
    for item in bucket.objects.all():
        path, filename = os.path.split(item.key)

        # create dir
        dir_path = save_path + '/' + path
        os.makedirs(dir_path, exist_ok=True)

        # download file
        bucket.download_file(item.key, dir_path + '/' + filename)
    print('Finished downloading in {} seconds.'.format(curr_time - time.time()))


if __name__ == '__main__':
    b = get_s3_bucket('cat-detector-dataset')
    download_bucket(b, 'dataset')