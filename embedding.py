import sys
import os
import yaml
import argparse
import numpy as np
from inception_resnet_v1 import *
from function import extract_face, get_embedding
import boto3
import uuid
import tempfile

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default='dinda@isi.co.id', help='userid (email) and folder name')
    return parser.parse_args()

args = get_args()
config = yaml.load(open('config.yaml'))

session = boto3.Session(profile_name=config['profile_name'])
dynamodb = session.resource('dynamodb')
table = dynamodb.Table(config['table_users'])

s3 = boto3.resource('s3')
bucket_name = config['bucket_name']
email = args.email
my_bucket = s3.Bucket(bucket_name)

facenet_model = InceptionResNetV1()
print("model built")

facenet_model.load_weights(config['model_dir'])
print('Loaded Model')

X, faces = list(), list()

count = 0
count_photo = 0
# get photos in s3 bucket
for my_bucket_object in my_bucket.objects.all():
    if email in my_bucket_object.key:
        try:
            count +=1

            if (count % config['photo_skip'] == 0):
                count_photo +=1
                print(my_bucket_object)
                object = my_bucket.Object(my_bucket_object.key)
                tmp = tempfile.NamedTemporaryFile()

                with open(tmp.name, 'wb') as f:
                    object.download_fileobj(f)
                    face = extract_face(tmp.name)
                    face = get_embedding(facenet_model, face)
                    faces.append(face)
        except Exception:
            count -=1
            pass

print("loaded %d sample for class: %s" % (len(faces),email) ) # print progress
X.extend(faces)
arr = np.array(X)
encoded = arr.tostring()


# insert data face to dynamodb
insert_data = table.put_item(
    Item={
        'id': str(uuid.uuid4()),
        'userid': email,
        'embeddingFace': encoded,
        'countPhoto': count_photo
    }
)

print(insert_data)