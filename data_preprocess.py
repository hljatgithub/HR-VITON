import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np
import json
import torchvision.transforms as transforms

def get_parse_agnostic(image_name, file_path): # 画image-parse-agnostic-v3.2
    fine_width, fine_height = 768, 1024

    pose_name = image_name+'_keypoints.json'
    with open(file_path+f'/openpose_json/{pose_name}', 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

    parse = Image.open(file_path+f'/image-parse-v3/{image_name}_gray.png') # PIL.Image 原读取image-parse-agnostic-v3.2
    parse = transforms.Resize(fine_width, interpolation=0)(parse) # import torchvision.transforms as transforms
    
    #parse_agnostic = get_parse_agnostic(parse, pose_data) # 这步跟源代码不一样，这版结合pose_data临时画图

    parse_array = np.array(parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (fine_width, fine_height), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm) # PIL.ImageDraw
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


#file_path = '/content/drive/MyDrive/AI/models/HR_VITOTN/data/total_data'
file_path = '/content/HR-VITON/HR-VITON-main/data/total_data'


os.chdir('/content/HR-VITON')
# 衣服图片处理···············································································
cloth_file_path = file_path + r'/cloth'
cloth_list = os.listdir(cloth_file_path)

for i,file in enumerate(cloth_list):
    print(i, len(cloth_list), file)
    # 包含文件类型结尾.jpg
    cloth_path = f'{cloth_file_path}/{file}'
    cloth_name = file.split('.')[0]
    # 1、检查分辨率············································
    print(cloth_path,'检查分辨率')
    ori_cloth = cv2.cvtColor(cv2.imread(cloth_path), cv2.COLOR_BGR2RGB)
    if ori_cloth.shape != (1024, 768, 3):
        cloth = cv2.imread(cloth_path)
        cloth = cv2.resize(cloth, (768, 1024))
        cv2.imwrite(cloth_path, cloth)
        print('已重置图片大小')

    # 2、输出衣服mask·············································
    output_path = file_path + f'/cloth-mask/{cloth_name}.jpg'
    print('生成衣服mask', output_path)
    terminnal_command = f"python get_cloth_mask.py --cloth_path {cloth_path} --output_path {output_path}" 
    os.system(terminnal_command)


# 人物图片处理···············································································
image_file_path = file_path + r'/image'
image_list = os.listdir(image_file_path)

for i,file in enumerate(image_list):
    print(i, len(image_list), file)
    # 包含文件类型结尾.jpg
    image_path = f'{image_file_path}/{file}'
    image_name = file.split('.')[0]
    # 1、检查分辨率·············································
    print(image_path,'检查分辨率')
    ori_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if ori_img.shape != (1024, 768, 3):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (768, 1024))
        cv2.imwrite(image_path, image)
        print('已重置图片大小')


    # 2、openpose_json：image_name_keypoints.json················
    # 包含文件类型结尾_keypoints.json
    output_path = file_path + f'/openpose_json/{image_name}_keypoints.json'
    print('生成keypoint.json', output_path)
    terminnal_command = f"python posenet.py --image_path {image_path} --output_path {output_path}" 
    os.system(terminnal_command)

    
    # 3、image-parse-v3：image_name.png·······················
    img=cv2.imread(image_path)
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    os.chdir('/content/HR-VITON/Graphonomy-master')
    # 不包含文件类型结尾,程序内部固定.png
    output_path = file_path + '/segmentation'
    print('生成人体部位图segmentation：image_name.png~用于去除原图背景，image_name_gray.png~未知')
    # terminnal_command =f"python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../image-parse-v3 --output_name /resized_segmentation_img"
    terminnal_command = f"python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path {output_path} --output_name /{image_name}"
    os.system(terminnal_command)

    # 读取上面输出的主体图
    print('通过segmentation图生成无背景图')
    mask_img=cv2.imread(output_path+f'/{image_name}.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    # 形态学转换~腐蚀erode，k为MORPH_RECT正方形核——有色区域变细，删掉更多背景
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    # 原图与主体图and部分保留得到原图主体
    img_seg=cv2.bitwise_and(ori_img, ori_img, mask=mask_img)
    # 原图和主体图作差得到背景
    back_ground=ori_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg) # 灰色背景
    # 原图无背景图
    #cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    output_path = file_path + '/image_no_background/{file}'
    #cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg',img)
    cv2.imwrite(output_path,img)
  
    print('segmentation图制作灰度图')
    image_path = file_path + f'/segmentation/{image_name}.png'
    output_path = file_path + f'/image-parse-v3/{image_name}.png'
    terminnal_command = f"python get_seg_grayscale.py --image_path {image_path} --output_path {output_path}"
    os.system(terminnal_command)
    
    
    # 4、image-parse-agnostic-v3.2：image_name.png···············
    agnostic = get_parse_agnostic(image_name, file_path)
    output_path = file_path + '/image-parse-agnostic-v3.2'
    cv2.imwrite(output_path, agnostic)


    # 5、姿态估计image-densepose：image_name.jpg
    #%cd /content/HR-VITON
    os.chdir('/content/HR-VITON')
    #terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    #https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    #origin.jpg --output output.pkl -v"
    # 输出data.json到/content/HR-VITON
    terminnal_command = f"python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    input {image_path} --output output.pkl -v"
    os.system(terminnal_command)
    print('\n')
    # 通过data.json姿态关键点数据保留人体躯干部分image_name.jpg
    output_path = file_path + '/image-densepose'
    terminnal_command = f"python get_densepose.py --image_path {image_path} --output_path {output_path}"
    os.system(terminnal_command)

    print('\n')
