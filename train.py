# Some basic setup:
# Setup detectron2 logger
# python train.py --yaml=mask_rcnn_X_101_32x8d_FPN_3x.yaml --output=mask_rcnn_X_101
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    register_coco_instances("my_dataset_train", {}, "data/train.json", "data/train")
    register_coco_instances("my_dataset_val", {}, "data/val.json", "data/val")

    metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    # cfg.merge_from_file(args.yaml)
    cfg.merge_from_file(args.yaml)
    # cfg.MODEL.WEIGHTS = os.path.join('mask_rcnn_X_101_small_anchor', "model_final.pth")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{args.yaml}")
    # cfg.MODEL.WEIGHTS = os.path.join('mask_rcnn_X_101', "model_final_best.pth")
    cfg.OUTPUT_DIR = args.output
    # if you have pre-trained weight.
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1500

    # faster, and good enough for this toy data5set
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # build output folder
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()