from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


import base64
import cv2
from PIL import Image
import insightface
import onnxruntime
import cv2
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
import torch
from huggingface_hub import InferenceClient
import numpy as np
import os
from handle_basicsr_issue import replace_keyword_in_package
replace_keyword_in_package()
from basicsr.archs.srvgg_arch import SRVGGNetCompact



face_swapper = insightface.model_zoo.get_model('app/models/inswapper_128.onnx',
                                                            providers=onnxruntime.get_available_providers())

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path='app/models/realesr-general-x4v3.pth', model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

# Use GFPGAN for face enhancement
face_enhancer = GFPGANer(model_path='app/models/GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
face_analyser.prepare(ctx_id=0, det_size=(640, 640))


def generate_image_interface(request):
    """
    Render an interface to upload an image for face verification.
    """
    return render(request, 'generate_image.html')


def generate_image_from_diffusion(user_prompt, token="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"):
    """
    Generates an image from a text prompt using a diffusion-based model.

    Parameters:
    ----------
    user_prompt : str
        A text prompt describing the desired image to generate.
    token : str, optional
        An authentication token for accessing the diffusion model API. 
        Defaults to an empty string.

    Returns:
    -------
    numpy.ndarray
        The generated image as a NumPy array in BGR format, suitable for OpenCV operations.    
    """
    client = InferenceClient("black-forest-labs/FLUX.1-dev", token=token)
    generated_image = client.text_to_image(user_prompt)
    generated_image = np.array(generated_image)
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
    return generated_image


def get_face(img_data):
    """
    Identifies and returns the largest detected face in the given image data.

    Parameters:
    ----------
    img_data : Any
        The input image data to be analyzed. The exact format and type of `img_data`
        depend on the `face_analyser.get` method, which processes and returns detected faces.

    Returns:
    -------
    largest : Object
        The data corresponding to the largest detected face in the image. The returned object
        is one of the entries produced by `face_analyser.get` and contains information
        about the detected face, including its bounding box (`bbox`).
    """
    analysed = face_analyser.get(img_data)
    largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return largest


def swap_generated_image(generated, generated_face, user_img):
    """
    Swaps a face from a generated image onto a user-provided image.

    Parameters:
    ----------
    generated : Any
        The generated image containing the face to be swapped. 
        The exact format depends on the `face_swapper.get` method.
    generated_face : Any
        The detected face data from the generated image. 
        This is typically a region or bounding box containing the face.
    user_img : Any
        The target user image where the face will be swapped.

    Returns:
    -------
    Any
        The resulting image with the face swapped. 
        The exact return type depends on the `face_swapper.get` method.
    """
    return face_swapper.get(generated, generated_face, user_img, paste_back=True)


def enhace_image(swapped_image):
    """
    Enhances a face-swapped image for improved quality and details.

    Parameters:
    ----------
    swapped_image : Any
        The input image containing the swapped face to be enhanced.
        The format must be compatible with the `face_enhancer.enhance` method.

    Returns:
    -------
    Any
        The enhanced image with better quality and details. 
        The exact return type depends on the `face_enhancer.enhance` method.
    """
    _, _, enhanced_swapped_image = face_enhancer.enhance(swapped_image,has_aligned=False, 
                                                only_center_face=False, paste_back=True)
    return enhanced_swapped_image


def image_to_base64(image):
    """
    Converts an image to a Base64-encoded string.

    Parameters:
    ----------
    image : PIL.Image.Image or numpy.ndarray
        The input image to be converted. If the input is a PIL image, it is first converted to a NumPy array.
        The image must be in RGB format if it has three channels.

    Returns:
    -------
    str
        A Base64-encoded string representation of the image in PNG format.
    """
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image)
    byte_data = buffer.tobytes()
    base64_image = base64.b64encode(byte_data).decode('utf-8')
    return base64_image


@csrf_exempt
def generate_image(request):
    """
    Handles an HTTP POST request to generate and optionally modify an image based on a user prompt.

    This function uses a diffusion model to generate an image from a text prompt. Optionally, 
    if a reference image is provided, it performs face swapping and enhances the resulting image. 
    The final image is returned as a Base64-encoded string in the JSON response.

    Parameters:
    ----------
    request : HttpRequest
        The HTTP request object containing the POST data.

        Expected POST data:
        - `user_prompr` (str): A text prompt for the diffusion model to generate an image.
        - `Image_File` (str, optional): Path or reference to a user-provided image for face swapping.

    Returns:
    -------
    JsonResponse
        A JSON response containing either:
        - `image` (str): A Base64-encoded string of the resulting image, or
        - `error` (str): An error message if the process fails.
    """
    try:
        user_prompt = request.POST.get('user_prompr', '')
        frame = generate_image_from_diffusion(user_prompt)        
        try:
            image_ = request.POST.get('image_path', '')
            face = get_face(frame)
            source_face = get_face(cv2.imread(image_))
            swapped_image = swap_generated_image(frame, face, source_face)
            result = enhace_image(swapped_image)
        except:
            result = frame
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = image_to_base64(result)
        return JsonResponse({'image': result})

    except Exception as e:
        # Handle any exceptions (If the image is not readable)
        return JsonResponse({'error': str(e)}, status=500)
