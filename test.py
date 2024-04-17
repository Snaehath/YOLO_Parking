import requests

# URL for the endpoint you want to call
url = 'http://localhost:5000/detect'

# Path to the image file you want to upload
image_file_path = './uploads/IMG_20240408_110950.jpg'

# Open the image file in binary mode
files = {'file': open(image_file_path, 'rb')}

# Make the POST request
response = requests.post(url, files=files)

# Print the response
print(response.text)