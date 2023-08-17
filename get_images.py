from requests import get
from datetime import datetime

def get_images(
    DATA_DIRECTORY = 'sample',
    INPUT_WIDTH = 320,
    INPUT_HEIGHT = 240):
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
        image_name = camera['camera_id'] + '_' + \
            datetime.now().strftime('%d%m%y%H%M') + '.jpg'
        with open('/'.join([DATA_DIRECTORY, image_name]), 'wb') as f:
            f.write(res.content)

if __name__ == 'main':
    get_images()