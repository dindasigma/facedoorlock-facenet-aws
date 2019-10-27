import numpy as np
import cv2
import sys
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from function import extract_face, get_embedding
import boto3

config = yaml.load(open('config.yaml'))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emdTrainX, trainy = list(), list()


session = boto3.Session(profile_name=config['profile_name'])
dynamodb = session.resource('dynamodb')
table = dynamodb.Table(config['table_users'])
response = table.scan()
peoples = len(response['Items'])

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

emdTrainX = np.asarray(emdTrainX)	
trainy = np.asarray(trainy)	
print(emdTrainX.shape, trainy.shape)

# normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)

print("trainy:")
print(trainy.shape)
print("encoder:")
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
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: #discard small detected faces
			cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224
			print('detected_face')
			print(detected_face)
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
				for i in range(peoples):
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
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()
