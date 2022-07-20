from imread import imread_from_blob
from requests import get
from datetime import time, datetime

DATA_DIRECTORY = 'sample'
INPUT_WIDTH, INPUT_HEIGHT = 320, 240

curr_time = datetime.now().time()
if curr_time < time(7, 0) or curr_time > time(19, 0):
    TIME = 'night'
else:
    TIME = 'day'

res = get('https://api.data.gov.sg/v1/transport/traffic-images')
data = res.json()

if not data['api_info']['status'] == 'healthy':
    print('Traffic API unavailable')
    exit()

for i in range(len(data['items'][0]['cameras'])):
    camera = data['items'][0]['cameras'][i]
    width, height = camera['image_metadata']['width'], \
        camera['image_metadata']['height']

    res = get(camera['image'])
    image_name = camera['camera_id'] + '.jpg'
    with open('/'.join([DATA_DIRECTORY, TIME, image_name]), 'wb') as f:
        f.write(res.content)
