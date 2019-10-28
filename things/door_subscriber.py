import yaml
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from time import sleep
import RPi.GPIO as GPIO # import Raspberry Pi GPIO library
import logging
import boto3
import json
import uuid
import time
import datetime

GPIO.setwarnings(True) # ignore warning
GPIO.setmode(GPIO.BOARD) # use physical pin numbering

config = yaml.load(open('config.yaml'))

session = boto3.Session(profile_name=config['profile_name'])
dynamodb = session.resource("dynamodb")
table_doors = dynamodb.Table(config['table_doors'])
table_log = dynamodb.Table(config['table_log'])


# callback from AWS subscribe
def doors_callback(client, userdata, message):
	print("\nReceived message: ")
	print(message.payload)
	print("From topic: ")
	print(message.topic)
	thing_switch(message)
	print("-----------------------------------\n")

def create_log(table_log, data):
	try:
		time = datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')

		response_insert = table_log.put_item(
			Item={
				'id': str(uuid.uuid4()),
				'userid': data['requester'],
				'doorid': data['pin'],
				'activity': 'open the door',
				'createdAt': time,
			}
		)
	except ClientError as e:
				return (e.response['Error']['Message'])
	else:
		return ("Recorded")


def thing_switch(message):
	data = json.loads(message.payload)

	# GPIO pin initialization
	pin = int(data["pin"])
	print(data["pin"])

	if data and 'command' in data:

		# turn on the relay if command is "open"
		if data["command"] == "open":
			GPIO.output(pin, GPIO.HIGH)
			sleep(5)
			print("After sleep ...")
			GPIO.output(pin, GPIO.LOW)
			print("[INFO] The door is open!")

			# create log
			create_log(table_log, data)
		else:
			print("[ERRO] Unknown command!")

# activating gpio pin
response_scan = table_doors.scan()
for i in response_scan["Items"]:
	GPIO.setup(int(i["pin"]), GPIO.OUT)
	GPIO.output(int(i["pin"]), GPIO.LOW)
	print("Activating pin: " + str(i["pin"]))

# configure loggger
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# AWS IoT client initialization ...
thing = AWSIoTMQTTClient(config['client_id'])
thing.configureEndpoint(config['endpoint'], config['port'])
thing.configureCredentials(config['root_ca_path'], config['private_key_path'], config['cert_path'])

thing.configureOfflinePublishQueueing(-1)
thing.configureDrainingFrequency(2)
thing.configureConnectDisconnectTimeout(10)
thing.configureMQTTOperationTimeout(5)

try:
	thing.connect()
	thing.subscribe(config['topic'], 1, doors_callback)

	while True:
		 """ I'm loop """

except KeyboardInterrupt:
	print("Cleaning up GPIO")
	GPIO.cleanup()
	thing.unsubscribe(config['topic'])

finally:
	print("Cleaning up GPIO")
	GPIO.cleanup()
	thing.unsubscribe(config['topic'])

