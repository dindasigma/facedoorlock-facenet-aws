# FaceDoorLock System with FaceNet and Amazon Web Services (AWS)

FaceDoorLock System implemented with AWS IoT Things, Amazon S3, and DynamoDB. Face detection was performed by using [Multi-task Cascaded Convolutional Networks (MTCNN)](https://arxiv.org/pdf/1604.02878.pdf) and face recognition by using [FaceNet](https://arxiv.org/abs/1503.03832) and Support Vector Machine (SVM).

## TODO List

1. Get people photos from Amazon S3 [done!]
2. Get Embedding face (using MTCNN and FaceNet) and save to DynamoDB [done!]
3. Detect face in realtime using MTCNN [done!]
3. Recognize face in realtime using FaceNet according to embedding face in DynamoDB  [done!]
4. Publish the name of detected face to the topic using AWS IoT Things and MQTT [done!]
6. **Create Log and save to DynamoDB (on subscriber) [todo**]



## Running Environment

- python 3.6
- awscli 1.16
- boto3 1.9
- keras 2.2
- mtcnn 0.0.9
- numpy 1.17
- scikit-learn 0.20 
- AWSIoTPythonSDK 1.4



## Pretrained Model
You can find pre-trained weights of 30 hours training with GPU and work on keras [here](https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view).

## Extract Embedding with Pretrained Model and Save to DynamoDB

You can extract embedding from face images with [embedding.py](https://github.com/dindasigma/facedoorlock-facenet-aws/blob/master/embedding.py) by the following script:

```
python embedding.py --email=dinda@isi.co.id
```

where email is the name of AWS S3 folder also index key of saved data in DynamoDB.

## Realtime face detection and recognition
Realtime face detection and recognition with [realtime.py](https://github.com/dindasigma/facedoorlock-facenet-aws/blob/master/realtime.py):
```
python realtime.py
```