import urllib.request
import json

from PIL import Image

def predict_from_api(image_path):
  url = "https://aircraft-classifier-torch.herokuapp.com/predict"
  req_header = {
      'Content-Type': 'application/json',
  }
  image = Image.open(image_path)
  req_data = json.dumps({
    "image": image
  })

  req = urllib.request.Request(url, data=req_data.encode(), method='POST', headers=req_header)
  with urllib.request.urlopen(req) as response:
    body = json.loads(response.read())
    headers = response.getheaders()
    status = response.getcode()
    print(body)
