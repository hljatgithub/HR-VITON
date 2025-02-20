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
    parser.add_argument('--cloth_path', type=str, default=True, help='cloth_path')
    parser.add_argument('--original_path', type=str, default=True, help='cloth_path')
    opt = parser.parse_args()

    # Get mask of cloth
    # 衣服原图和黑白蒙版图另存为
    print("1、获取服装mask图···············\n")
    # cloth_path = input('1、衣服图片路径：')
    cloth_path = opt.cloth_path
    terminnal_command = f"python get_cloth_mask.py --cloth_path {cloth_path} --output_path ./HR-VITON-main/test/test/cloth-mask/00001_00.jpg" 
    os.system(terminnal_command)
    
    # Read input image # 模特原图
    #original_path = input('2、模特图片路径：')
    original_path = opt.original_path
    img=cv2.imread(original_path)
    ori_img=cv2.resize(img,(768,1024))
    cv2.imwrite("./origin.jpg",ori_img)

    # Resize input image # 模特图重置大小给Graphonomy部位识别
    img=cv2.imread('origin.jpg')
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)

    # Get openpose coordinate using posenet
    # 生成关键点数据00001_00_keypoints.json，修改了输入参数
    print("3、posenet生成keypoints数据················\n")
    terminnal_command = f"python posenet.py --image_path {original_path} --output_path ./HR-VITON-main/test/test/openpose_json/00001_00_keypoints.json --model test" 
    os.system(terminnal_command)

    # Generate semantic segmentation using Graphonomy-Master library
    # 人物部位识别分割、主体识别输出resized_segmentation_img
    print("4、基于Graphonomy-Master的DeepLabv3+模型生成segmentation图···············\n")
    os.chdir("./Graphonomy-master")
    terminnal_command = "python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img --model test"
    os.system(terminnal_command)
    os.chdir("../")

    # Remove background image using semantic segmentation mask
    # 读取上面输出的主体图
    print('5、通过segmentation图生成无背景图························\n')
    mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    # 形态学转换~腐蚀erode，k为MORPH_RECT正方形核——有色区域变细，删掉更多背景
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    # 原图与主体图and部分保留得到原图主体
    img_seg=cv2.bitwise_and(ori_img,ori_img,mask=mask_img)
    # 原图和主体图作差得到背景
    back_ground=ori_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg) # 灰色背景
    # 原图无背景图
    cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg',img)

    # Generate grayscale semantic segmentation image
    # 用主题图resized_segmentation_img制作灰度图
    print('6、通过segmentation图生成灰度图························\n')
    terminnal_command =f"python get_seg_grayscale.py --image_path ./resized_segmentation_img.png --output_path ./HR-VITON-main/test/test/image-parse-v3/00001_00.png"
    os.system(terminnal_command)

    # Generate Densepose image using detectron2 library
    # detectron2可以做多种识别,densepose姿态估计输出data.json
    print("7、通过detectron2生成densepose姿态估计························\n")
    # 加载设置默认参数,生成data.json
    terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    origin.jpg --output output.pkl -v"
    os.system(terminnal_command)
    # 通过data.json姿态关键点数据保留人体躯干部分，修改了输入参数
    terminnal_command = f"python get_densepose.py --image_path {original_path} --output_path ./HR-VITON-main/test/test/image-densepose/00001_00.jpg"
    os.system(terminnal_command)

    # Run HR-VITON to generate final image
    print("8、最终通过HR-VITON处理输出结果························\n")
    os.chdir("./HR-VITON-main")
    # data_list~每行是一对image cloth二元组对,空格隔开文件名
    terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
    os.system(terminnal_command)

    # Add Background or Not
    l=glob.glob("./Output/*.png") # 00001_00_00001_00.png
    print(l)
    # 批量导出——上面代码输出命名需改代码，data_list文件写入代码缺失
    # Add Background
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

    # Remove Background
    else:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img,img,mask=mask_img)
            cv2.imwrite(i,img)


    os.chdir("../")
    cv2.imwrite("./static/finalimg.png", img)
