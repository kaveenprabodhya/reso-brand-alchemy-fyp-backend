from openai import OpenAI
from datetime import datetime
import requests
import os
import uuid

client = OpenAI()


def generate_image_from_text(brand_name, colors_captured):
    colors_string = ', '.join(colors_captured)
    description = f'''Create a logo named {brand_name}, specially harmonizing colors {colors_string} with other colors. Use any icon related to music to beautify the logo, and do not lose brand name characters.'''
    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=description,
            size="512x512",
            quality="standard",
            n=5,
        )
        img_url_array = []
        for img_data in response.data:
            img_url_array.append(img_data.url)

        print(img_url_array)
        return img_url_array
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def download_images(img_url_array, batch_id):
    img_paths = []
    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist

    for idx, url in enumerate(img_url_array):
        img_data = requests.get(url).content
        img_path = f'{directory}/image_{batch_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        with open(img_path, 'wb') as handler:
            handler.write(img_data)
        img_paths.append(img_path)
    return img_paths
