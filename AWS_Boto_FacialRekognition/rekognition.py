import boto3
import requests
from pprint import pprint

def get_img_from_url(imgurl):
    resp = requests.get(imgurl)
    imgbytes = resp.content
    return imgbytes

def get_img_from_folder(filename):
    with open(filename, 'rb') as imgfile:
        return imgfile.read()


url = 'https://www.extremetech.com/wp-content/uploads/2019/12/SONATA-hero-option1-764A5360-edit.jpg'

client = boto3.client('rekognition')
img = get_img_from_url (url)
# Image is a map, therefore use {}
rekresp = client.detect_labels(Image ={'Bytes' : img}, MinConfidence = 30)
pprint(rekresp)


for label in rekresp['Labels']:
    print (label ['Name'], label ['Confidence'])
