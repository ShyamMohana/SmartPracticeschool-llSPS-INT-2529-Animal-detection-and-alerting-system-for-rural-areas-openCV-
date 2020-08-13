#Visual recognition
import cv2
import numpy as np
import datetime
import json
from watson_developer_cloud import VisualRecognitionV3
import time

#sending data to ibm iot from python
import time
import sys
import ibmiotf.application
import ibmiotf.device

#Text to speech
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from playsound import playsound

#ObjectStorage
import ibm_boto3
from ibm_botocore.client import Config, ClientError

#CloudantDB
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
import requests

#Provide your IBM Watson Device Credentials
organization = "i2a1nf"
deviceType = "raspberrypi1"
deviceId = "123456"
authMethod = "token"
authToken = "0123456789"

def myCommandCallback(cmd):
        print("Command received: %s" % cmd.data)#Commands
try:
	deviceOptions = {"org": organization, "type": deviceType, "id": deviceId, "auth-method": authMethod, "auth-token": authToken}
	deviceCli = ibmiotf.device.Client(deviceOptions)
	#..............................................
	
except Exception as e:
	print("Caught exception connecting device: %s" % str(e))
	sys.exit()
deviceCli.connect()

#Visual recognition service credentials
visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey="kpo-llZt-6MsZaARmfF_LkJyiXa6cVDafXshPxv-JR68")

#haarcascade code for face detection
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier=cv2.CascadeClassifier("haarcascade_eye.xml")

#text to speech credentials
authenticator = IAMAuthenticator('DIuG81FMMWMlUfeupPISj2zw-sYu8r5BntItJqrNs-BF')
text_to_speech = TextToSpeechV1(
   authenticator=authenticator
   )
text_to_speech.set_service_url('https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/25ef33ad-e64f-4dc9-a4d4-c79395abc8dd')

# Constants for IBM COS values
COS_ENDPOINT = "https://s3.jp-tok.cloud-object-storage.appdomain.cloud" 
COS_API_KEY_ID = "-M8l90xGZCPLgwRMK9gW8a9wDcuXg1jpkAUGeh68YYL3"
COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/identity/token"
COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/2515ad9f36414fb6ad74269c08b82f8e:980685ff-0708-4f3a-b541-6f543c0b435d::"

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

#cloudantdB credentials
client = Cloudant("07abcd4a-109f-4cbe-960f-fb557187a4af-bluemix", "c9fd0d50a3eff80289cbafce07d66528521cfd22cc7ea2105134acdf489377fe", url="https://07abcd4a-109f-4cbe-960f-fb557187a4af-bluemix:c9fd0d50a3eff80289cbafce07d66528521cfd22cc7ea2105134acdf489377fe@07abcd4a-109f-4cbe-960f-fb557187a4af-bluemix.cloudantnosqldb.appdomain.cloud")
client.connect()

#database name
database_name = "animaldetection"

#creating database
my_database = client.create_database(database_name)

#checking whether the database exists or not
if my_database.exists():
   print(f"'{database_name}' successfully created.")

#text to speech conversion
with open('project.mp3', 'wb') as audio_file:
              audio_file.write(
                 text_to_speech.synthesize(
                    'Animal is detected. People Please be Safe and Cautious!!!',
                    voice='en-US_AllisonVoice',
                    accept='audio/mp3').get_result().content)

#publish data to IBM wastson platform
def myOnPublishCallback():
            print ("Published Data to IBM Watson Platform")

#Uploading the detected animal image to database
def multi_part_upload(bucket_name, item_name, file_path):
    try:
        print("Starting file transfer for {0} to bucket: {1}\n".format(item_name, bucket_name))
        # set 5 MB chunks
        part_size = 1024 * 1024 * 5

        # set threadhold to 15 MB
        file_threshold = 1024 * 1024 * 15

        # set the transfer threshold and chunk size
        transfer_config = ibm_boto3.s3.transfer.TransferConfig(
            multipart_threshold=file_threshold,
            multipart_chunksize=part_size
        )

        # the upload_fileobj method will automatically execute a multi-part upload
        # in 5 MB chunks for all files over 15 MB
        with open(file_path, "rb") as file_data:
            cos.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data,
                Config=transfer_config
            )

        print("Transfer for {0} Complete!\n".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to complete multi-part upload: {0}".format(e))


#It will read the first frame/image of the video
video=cv2.VideoCapture(0)

while True:
    #capture the first frame
    check,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #detect the faces from the video using detectMultiScale function
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    eyes=eye_classifier.detectMultiScale(gray,1.3,5)
    print(faces)#gives the pixel values for the detected face
    print(eyes)#gives the pixel values for eyes
    
    #drawing rectangle boundries for the detected face
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('animaldetection', frame)

        #saving the picture with date and time
        picname=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        cv2.imwrite(picname+".jpg",frame)

        #Sending the captured picture to Visual Recognition Service
        with open(picname+".jpg", 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                classifier_ids='DefaultCustomModel_298180192').get_result()

        #Gets the class and confidence score of the given picture
        text = "The image belongs to the class " + classes['images'][0]['classifiers'][0]['classes'][0]['class'] + ". The confidence level is " + str(classes['images'][0]['classifiers'][0]['classes'][0]['score'])
        print("\n" + text)

        #compares the class name with forestanimal 
        if((classes['images'][0]['classifiers'][0]['classes'][0]['class']=="forestanimals" or classes['images'][0]['classifiers'][0]['classes'][0]['class']=="forestanimal")and(classes['images'][0]['classifiers'][0]['classes'][0]['score']>=0.87)):

           #uploads the identified forestanimal picture to cloudantdB
           multi_part_upload("animaldetection", picname+".jpg", picname+".jpg")
           json_document={"link":COS_ENDPOINT+"/"+"animaldetection"+"/"+picname+".jpg"}
           new_document = my_database.create_document(json_document)

           #Publish data to IBM watson
           data="Animal is detected.People please be safe and cautious!!!"
           success = deviceCli.publishEvent("animal_detection", "json", data, qos=0, on_publish=myOnPublishCallback)
           if not success:
                   print("Not connected to IoTF")

           #Generates voice commands to alert the people
           playsound('project.mp3')

           # Check that the document exists in the database.
           if new_document.exists():
               print(f"Document successfully created.") 

        #drawing rectangle boundries for the detected eyes
        for(ex,ey,ew,eh) in eyes:
           cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (127,0,255), 2)
           cv2.imshow('animaldetection', frame)

        #waitKey(1)- for every 1 millisecond new frame will be captured
        Key=cv2.waitKey(1)
        if Key==ord('q'):
           #release the camera
           video.release()
           #destroy all windows
           cv2.destroyAllWindows()
           break

# Disconnect the device and application from the cloud
deviceCli.disconnect()
