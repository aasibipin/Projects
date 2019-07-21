import urllib.request
import json


weather_api = "efd4253984a5d6d2f3edd53b754a12c0"
url = "http://api.openweathermap.org/data/2.5/weather?q=Toronto,CA&units=metric&appid=%s"%weather_api 

response = urllib.request.urlopen(url).read()

json_obj = str(response, 'utf-8')
data = json.loads(json_obj)

