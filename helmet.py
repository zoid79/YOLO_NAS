from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    import sys

    stdout = sys.stdout
    import torch
    from super_gradients.training import Trainer

    from super_gradients.training import dataloaders
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train,
        coco_detection_yolo_format_val
    )

    from super_gradients.training import models

    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import (
        DetectionMetrics_050,
        DetectionMetrics_050_095
    )
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    sys.stdout = stdout

    from tqdm.auto import tqdm

    import os
    import requests
    import zipfile
    import cv2
    import matplotlib.pyplot as plt
    import glob
    import numpy as np
    import random

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    ROOT_DIR = 'Hard-Hat-Universe-26'
    train_imgs_dir = 'train/images'
    train_labels_dir = 'train/labels'
    val_imgs_dir = 'valid/images'
    val_labels_dir = 'valid/labels'
    test_imgs_dir = 'test/images'
    test_labels_dir = 'test/labels'
    classes = ['head', 'helmet', 'hi-viz helmet', 'hi-viz vest', 'person']
    print("start")

    dataset_params = {
        'data_dir':ROOT_DIR,
        'train_images_dir':train_imgs_dir,
        'train_labels_dir':train_labels_dir,
        'val_images_dir':val_imgs_dir,
        'val_labels_dir':val_labels_dir,
        'test_images_dir':test_imgs_dir,
        'test_labels_dir':test_labels_dir,
        'classes':classes
    }


    EPOCHS = 5
    BATCH_SIZE = 16
    print("start")

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 8
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 8
        }
    )
    print("start")

    train_params = {
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50:0.95'
    }

    model_to_train = 'yolo_nas_s'
    CHECKPOINT_DIR = 'checkpoints'

    trainer = Trainer(
        experiment_name=model_to_train,
        ckpt_root_dir=CHECKPOINT_DIR
    )
    print("start")

    model = models.get(
        model_to_train,
        num_classes=len(dataset_params['classes']),
        pretrained_weights="coco"
    )
    print("start")
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )

