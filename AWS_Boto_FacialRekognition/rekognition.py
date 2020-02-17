import boto3
import requests
from pprint import pprint

client = boto3.client('rekognition')


def get_img_from_url(imgurl):
    resp = requests.get(imgurl)
    imgbytes = resp.content
    return imgbytes

def get_img_from_folder(filename):
    with open(filename, 'rb') as imgfile:
        return imgfile.read()

img = get_img_from_url ('https://scontent.fykz1-1.fna.fbcdn.net/v/t1.15752-9/70487821_1041642329339731_5096054670380498944_n.jpg?_nc_cat=103&_nc_oc=AQlP1hzzarUbRsKdS9Yw4YMT5XdJqNbOsw4-1OKomHOUBl2gHG7poH3iIYMEvWQswnA&_nc_ht=scontent.fykz1-1.fna&oh=5aa7a2ef3848ac80b711748fecaf551e&oe=5E3D29C9')

rekresp = client.detect_labels(Image ={'Bytes' : img}, MinConfidence = 30)             #Image is a map, therefore use {}

pprint(rekresp)

for label in rekresp['Labels']:
    print (label ['Name'], label ['Confidence'])


