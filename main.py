import os, sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse
from cloths_segmentation.pre_trained_models import create_model

# 当前路径项目HR-VITON内

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    # Get mask of cloth
    # 衣服原图和黑白蒙版图另存为
    print("Get mask of cloth\n")
    terminnal_command = "python get_cloth_mask.py" 
    os.system(terminnal_command)
    
    # Read input image # 模特原图
    original_path = input('模特图片路径：')
    img=cv2.imread(original_path)
    ori_img=cv2.resize(img,(768,1024))
    cv2.imwrite("./origin.jpg",ori_img)

    # Resize input image # 模特图重置大小给Graphonomy部位识别
    img=cv2.imread('origin.jpg')
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)

    # Get openpose coordinate using posenet
    # 生成关键点数据00001_00_keypoints.json给Graphonomy部位识别
    print("Get openpose coordinate using posenet\n")
    terminnal_command = "python posenet.py" 
    os.system(terminnal_command)

    # Generate semantic segmentation using Graphonomy-Master library
    # 人物部位识别、主体识别输出resized_segmentation_img
    print("Generate semantic segmentation using Graphonomy-Master library\n")
    os.chdir("./Graphonomy-master")
    terminnal_command ="python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img"
    os.system(terminnal_command)
    os.chdir("../")

    # Remove background image using semantic segmentation mask
    # 读取上面输出的主体图
    mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    # 原图和主体图作差得到背景
    img_seg=cv2.bitwise_and(ori_img,ori_img,mask=mask_img)
    back_ground=ori_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg)
    # 原图无背景图
    cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg',img)

    # Generate grayscale semantic segmentation image
    # 用主题图resized_segmentation_img制作灰度图
    terminnal_command ="python get_seg_grayscale.py"
    os.system(terminnal_command)

    # Generate Densepose image using detectron2 library
    # detectron2可以做多种识别,densepose姿态估计
    print("\nGenerate Densepose image using detectron2 library\n")
    terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    origin.jpg --output output.pkl -v"
    os.system(terminnal_command)
    # 通过data.json姿态关键点数据保留人体躯干部分
    terminnal_command ="python get_densepose.py"
    os.system(terminnal_command)

    # Run HR-VITON to generate final image
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON-main")
    terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
    os.system(terminnal_command)

    # Add Background or Not
    l=glob.glob("./Output/*.png")

    # Add Background
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img,img,mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i,img)

    # Remove Background
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

    os.chdir("../")
    cv2.imwrite("./static/finalimg.png", img)
