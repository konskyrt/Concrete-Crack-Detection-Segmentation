
import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from .cv2_utils import getContours
import torchvision.transforms as transforms
from .models.deepcrack_model import DeepCrackModel

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)

def read_image(bytesImg, dim=(256, 256)):    #Decode Bytes to array
    img_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
    img = np.fromstring(bytesImg, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # adjust the image size
    w, h = dim
    if w > 0 or h > 0:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # apply the transform to both A and B
    img = img_transforms(Image.fromarray(img.copy()))   
    return img    

def create_model(opt, cp_path='pretrained_net_G.pth'):
    model = DeepCrackModel(opt)      # create a model given opt.model and other options
    checkpoint = torch.load(cp_path)
    if hasattr(model.netG, 'module'):
        model.netG.module.load_state_dict(checkpoint, strict=False)
    else:
        model.netG.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color = (255, 0, 0),
    alpha: float = 0.5, 
    resize = (256, 256)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(1, 1, 3)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=2)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv2.resize(image, resize)
        image_overlay = cv2.resize(image_overlay, resize)
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined

def inference(model, bytesImg, dim, unit):
    #print(img_path) 
    
    image = read_image(bytesImg) #Read Array
    # batchify
    image = image.unsqueeze(0)
    # hacky way to pass ground truth label
    model.set_input({'image': image, 'label': torch.zeros_like(image), 'A_paths':''}) 
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    confidence = visuals['fused'].max()

    # fused for final prediction
    for key in visuals.keys():
        visuals[key] = tensor2im(visuals[key])
        
    h, w, _ = visuals['fused'].shape
    fused = Image.fromarray(visuals['fused'])
    fused = np.array(fused, dtype='uint8')
    realHeight=dim[1]
    realWidth=dim[0]

    mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    mask[mask < 90] = 0
    mask[mask >= 90] = 255
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    
    overlay_img = overlay(tensor2im(image), mask, alpha=0)
    cv2.drawContours(image=overlay_img, contours=cnts[0], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    contour_img = getContours(fused, overlay_img, realHeight, realWidth, unit, confidence)

    return contour_img if contour_img is not None else overlay_img, visuals
    

    