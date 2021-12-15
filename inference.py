# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import numpy as np
import os
import json
import cv2
import random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import pycocotools.mask as mask_util
import argparse
setup_logger()


def instances_to_coco_json(instances, img_id):
    """
    Reference:
        detectron2 coco evaluator
        https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html
    Dump an "Instances" object to a COCO-format json that's used for evaluation

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k]+1,
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--model", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    register_coco_instances(
        "my_dataset_train",
        {},
        "data/train.json",
        "data/train"
    )
    metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    cfg.merge_from_file(args.yaml)
    cfg.MODEL.WEIGHTS = args.model  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    predictor = DefaultPredictor(cfg)
    image_dict = {
        "TCGA-A7-A13E-01Z-00-DX1.png": 1,
        "TCGA-50-5931-01Z-00-DX1.png": 2,
        "TCGA-G2-A2EK-01A-02-TSB.png": 3,
        "TCGA-AY-A8YK-01A-01-TS1.png": 4,
        "TCGA-G9-6336-01Z-00-DX1.png": 5,
        "TCGA-G9-6348-01Z-00-DX1.png": 6
    }
    test_files = os.listdir('data/test')
    results = []
    for i in test_files:
        print(i)
        im = cv2.imread(f'data/test/{i}')
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(len(outputs["instances"]))
        cv2.imshow('pred', out.get_image()[:, :, ::-1])
        cv2.waitKey(10)
        cv2.imwrite(i, out.get_image()[:, :, ::-1])
        results.extend(
            instances_to_coco_json(
                outputs["instances"].to("cpu"),
                image_dict[i])
        )
    cv2.destroyAllWindows()
    json_object = json.dumps(results, indent=4)
    with open("answer.json", "w") as outfile:
        outfile.write(json_object)
