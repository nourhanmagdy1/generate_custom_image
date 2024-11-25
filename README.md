# AI Image Generation and Face Swap Django App

## Overview

This Django application provides a simple interface for generating AI-based images from text prompts. Users can also optionally upload their own image to swap a face from the generated image onto the uploaded one, creating a personalized and enhanced result. 

## Features

- **AI Image Generation**: Generates high-quality images using a text-to-image diffusion model.
- **Face Swap Option**: Allows users to upload their own image, detecting faces in both images and swapping them seamlessly.
- **Image Enhancement**: Enhances the final image for better quality and detail.
- **Simple API Interface**: Accepts requests with a text prompt and optional image upload via HTTP POST, returning a Base64-encoded result.


### Installation

1. **Clone the Repository**:
   ```bash
   git clone customize_generation_system
   cd customize_generation_system
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   venv/Scripts/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server**:
   ```bash
   python manage.py runserver
   ```

**Parameters**:

- `user_prompr` (required): A text prompt describing the desired image to generate.
- `image_path` (optional): A file containing a user image for face swapping.

**Response**:

- `image`: Base64-encoded string of the generated/enhanced image.
- `error`: An error message if the process fails.

#### Example Request

**Using `requests` library**:
```python
import requests

url = "http://127.0.0.1:8000/"
data = {'user_prompr': "A serene landscape with mountains and a river"}
files = {'image_path': "A path of user image"}  # Optional

response = requests.post(url, data=data, files=files)
print(response.json())  # The response contains the Base64-encoded image or an error message.
```

## How It Works

1. **User Prompt**: The user provides a text description of the desired image.
2. **Image Generation**: The system generates an image using an AI diffusion model.
3. **Face Detection and Swapping**:
   - If a user image is provided, faces in both the generated and user-provided images are detected.
   - The face from the generated image is swapped onto the user-provided image.
4. **Enhancement**: The final image is enhanced for improved quality and detail.
5. **Base64 Encoding**: The result is encoded to Base64 and returned in a JSON response.

## Example Scenarios

- **Without Face Swap**:
  - Input: `{"user_prompr": "A futuristic robot in a neon-lit city"}`
  - Output: AI-generated image based on the text prompt.
  
- **With Face Swap**:
  - Input: 
    - `{"user_prompr": "A portrait of a renaissance queen"}`
    - `{"image_path": "A path of user image"}`
  - Output: The face from the generated portrait is swapped with the user's face.

## Customization

- Update the diffusion model in `generate_image_from_diffusion` if required.
- Integrate additional AI models for more customization options.
