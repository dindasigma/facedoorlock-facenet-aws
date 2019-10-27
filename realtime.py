import numpy as np
import cv2
import sys
import yaml
import json
from PIL import Image
from mtcnn.mtcnn import MTCNN

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from function import get_embedding
import boto3
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

def get_people():
	# Init DynamoDB
	session = boto3.Session(profile_name=config['profile_name'])
	dynamodb = session.resource('dynamodb')
	# select table from DynamoDB
	table = dynamodb.Table(config['table_users'])
	response = table.scan()
	count_people = len(response['Items'])

	emdTrainX, trainy = list(), list()

	for people in response['Items']:
		print(people['userid'])
		encoded = people['embeddingFace'].value
		encode_array = np.frombuffer(encoded, dtype='float32')
		i = int(people['countPhoto'])
		embds_arr = encode_array.reshape(i, 128)
		for face in embds_arr:
			emdTrainX.append(face)
		
		for x in range(i):
			trainy.append(people['userid'])

	return emdTrainX, trainy, count_people
	

config = yaml.load(open('config.yaml'))

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = AWSIoTMQTTClient(config['client_id'])
myAWSIoTMQTTClient.configureEndpoint(config['endpoint'], config['port'])
myAWSIoTMQTTClient.configureCredentials(config['root_ca_path'], config['private_key_path'], config['cert_path'])

# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()

emdTrainX, trainy, count_people = get_people()

emdTrainX = np.asarray(emdTrainX)	
trainy = np.asarray(trainy)	
print(emdTrainX.shape, trainy.shape)

# normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

from inception_resnet_v1 import *
facenet_model = InceptionResNetV1()
print("model built")

facenet_model.load_weights('facenet_weights.h5')
print("weights loaded")


cap = cv2.VideoCapture(0) #webcam

while(True):
	ret, img = cap.read()
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(img)
	print('results')
	#print(results)
	for i in range(len(results)):
		x, y, w, h = results[i]['box']
		
		if w > 130: #discard small detected faces
			cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224

			if detected_face is not None:
				
				random_face_emd = get_embedding(facenet_model, detected_face)

				# prediction for the face
				samples = np.expand_dims(random_face_emd, axis=0)
				yhat_class = model.predict(samples)
				yhat_prob = model.predict_proba(samples)
				# get name
				class_index = yhat_class[0]
				class_probability = yhat_prob[0,class_index] * 100
				predict_names = out_encoder.inverse_transform(yhat_class)

				label_name = 'unknown'
				labels = []
				for i in range(count_people):
					labels.append(i)

				all_names = out_encoder.inverse_transform(labels)


				index_max = np.argmax(yhat_prob[0])
				label_name = all_names[index_max]

				print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]))

				if index_max > config['threshold']:

					cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (67,67,67), 2)
							
					#connect face and text
					cv2.line(img,(x+w, y-64),(x+w-25, y-64),(67,67,67),1)
					cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),(67,67,67),1)

					# open the door
					message = {}
					message['pin'] = config['door_pin']
					message['command'] = 'open'
					message['requester'] = label_name
					messageJson = json.dumps(message)
					myAWSIoTMQTTClient.publish(config['topic'], messageJson, 1)
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()
