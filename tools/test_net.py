# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.engine.inference import inference
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir, save_config


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file", \
                        default="./configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml")
    parser.add_argument("--netname", default="mpprcnn", help="datetime of training", type=str)
    parser.add_argument("--date", help="datetime of training", type=str)
    parser.add_argument("--iter", help="iter of checkpoint", default=0, type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, \
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    output_dir = os.path.join(cfg.OUTPUT_DIR, args.netname, args.date + "/")
    cfg.OUTPUT_DIR = output_dir
    if args.iter == 0:
        cfg.MODEL.WEIGHT = os.path.join(output_dir, "model_final.pth")
    else:
        cfg.MODEL.WEIGHT = os.path.join(output_dir, "model_%07d.pth" % args.iter)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    output_config_path = os.path.join(output_dir, 'config_test.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    logger.info("fdasfasd" + cfg.MODEL.WEIGHT)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox", )
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm", )
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
