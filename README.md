# VRDL HW3

## 1. Install dependencies
Please follow the [detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install required dependences.

## 2. Training code
In this homework, I use [detectron2](https://github.com/facebookresearch/detectron2) developed by [facebookresearch](https://github.com/facebookresearch). First, we should convert training data to coco format. Finally, we can use coco format ddata to train our model.

```
python train.py \
--yaml=configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
--output=mask_rcnn_X_101_small_anchor/
```

* **--yaml:** the path to model yaml
* **--output:** the directory to store our model checkpoint

## 3. Inference code
The inference code will generate our result with coco format.

```
python inference.py \
--yaml=configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
--model=mask_rcnn_X_101_small_anchor/model_final.pth
```

* **--yaml:** the path to model yaml
* **--model:** the path to model checkpoint

## 4. Pre-trained models
You can download the [model weight](https://drive.google.com/file/d/1u5-chiYcklEZOeLc1geim7UgZRB-fil5/view?usp=sharing) to reproduce my result.

```
python inference.py \
--yaml=configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
--model=model_final_best_242.pth
```

## Result
![](https://i.imgur.com/97EiNq6.jpg)

## Reference
1. [detectron2 config](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml)
2. [instance to coco](https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html)
3. [detectron2 install](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
4. [covert mask to coco](https://github.com/vbnmzxc9513/Nuclei-detection_detectron2)