from imantics import Mask, Image, Category, Dataset
import cv2
import os
import sys
import json

print(sys.getrecursionlimit())
sys.setrecursionlimit(2000)
train_files = os.listdir('dataset/train')
image_list = []
dataset = Dataset('Data Science Bowl 2018')
for image_id, image_name in enumerate(train_files):
    print(image_name)
    image = Image.from_path(
        f'dataset/train/{image_name}/images/{image_name}.png'
    )
    image.id = image_id + 1
    mask_files = os.listdir(f'dataset/train/{image_name}/masks')
    for mask_name in mask_files:
        if mask_name.endswith('.png'):
            mask_array = cv2.imread(
                f'dataset/train/{image_name}/masks/{mask_name}',
                cv2.IMREAD_UNCHANGED
            )
            mask = Mask(mask_array)
            categ = Category("nucleus")
            categ.id = 1
            image.add(mask, category=categ)
    image_coco_json = image.coco()
    with open(f'coco/{image_name}.json', 'w') as output_json_file:
        json.dump(image_coco_json, output_json_file, indent=4)
    dataset.add(image)

coco_json = dataset.coco()
annotations = coco_json["annotations"]
correct_seg = []
for i in range(len(annotations)):
    if len(annotations[i]["segmentation"][0]) <= 4:
        print(annotations[i]["id"])
        continue
    correct_seg.append(annotations[i])
    coco_json["annotations"] = correct_seg
with open('train.json', 'w') as output_json_file:
    json.dump(coco_json, output_json_file, indent=4)
