from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cloth_path', type=str, default='', help='person image path')
parser.add_argument('--output_path', type=str, default='', help='000001_0_keypoints output path')
opt = parser.parse_args()
#cloth_path = input('衣服图片路径：')
cloth_path = opt.cloth_path
output_path = opt.output_path
    
    
model = create_model("Unet_2020-10-30")
model.eval()
image = load_rgb(cloth_path) # 长、宽、位深

transform = albu.Compose([albu.Normalize(p=1)], p=1)

# 图像边缘填充边框，border为填充方式——似乎目的为填充到能整除32
padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
    prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads) # 删除填充的外圈

img=np.full((1024,768,3), 255) # 全白
seg_img=np.full((1024,768), 0) # 黑底

b=cv2.imread(cloth_path)
b_img = mask* 255 # 全白
                 
if b.shape[1]<=600 and b.shape[0]<=500:
    b=cv2.resize(b,(int(b.shape[1]*1.2),int(b.shape[0]*1.2)))
    b_img=cv2.resize(b_img,(int(b_img.shape[1]*1.2),int(b_img.shape[0]*1.2)))
    
shape=b_img.shape
# 下面应该在统一大小
img[int((1024-shape[0])/2): 1024-int((1024-shape[0])/2),int((768-shape[1])/2):768-int((768-shape[1])/2)]=b # img白画布上放原图
seg_img[int((1024-shape[0])/2): 1024-int((1024-shape[0])/2),int((768-shape[1])/2):768-int((768-shape[1])/2)]=b_img # 黑画布放mask白图


cv2.imwrite(output_path.replace('cloth-mask', 'cloth'), img)
#cv2.imwrite("./HR-VITON-main/test/test/cloth-mask/00001_00.jpg",seg_img)
cv2.imwrite(output_path, seg_img)

