import os

from super_gradients.examples.quantization.resnet_qat_example import *
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import *
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import *
from super_gradients.training.models.detection_models.pp_yolo_e import *

data_dir = '../dataset'
train_imgs_dir = 'train/images'
train_lbls_dir = 'train/labels'
valid_imgs_dir = 'valid/images'
valid_lbls_dir = 'valid/labels'
test_imgs_dir = 'test/images'
test_lbls_dir = 'test/labels'
classes = ['head', 'helmet']
checkpoints_dir = '../checkpoints'

model_name = 'yolo_nas_s'

epochs = 50
batch_size = 16
num_workers = 8

optimizer = "ADAM"
optimizer_params = {"weight_decay": 0.0001}

warmup_mode = "LinearEpochLRWarmup"
warmup_initial_lr = 1e-6
lr_warmup_epochs = 3
initial_lr = 5e-4
lr_mode = "cosine"
cosine_final_lr_ratio = 0.1

metrics = [
    DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=len(classes),
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
        num_cls=len(classes),
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=0.7
        )
    )
]

loss_function = PPYoloELoss(
    use_static_assigner=False,
    num_classes=len(classes),
    reg_max=16
)

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    torch.multiprocessing.freeze_support()  # 윈도우 시스템의 재귀 Fork 시스템콜을 방지.
    train_data_loader = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': data_dir,
            'images_dir': train_imgs_dir,
            'labels_dir': train_lbls_dir,
            'classes': classes
        },
        dataloader_params={
            'batch_size': batch_size,
            'num_workers': num_workers
        }
    )

    valid_data_loader = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': data_dir,
            'images_dir': valid_imgs_dir,
            'labels_dir': valid_lbls_dir,
            'classes': classes
        },
        dataloader_params={
            'batch_size': batch_size,
            'num_workers': num_workers
        }
    )

    train_params = {
        'silent_mode': False,
        "average_best_models": True,
        "warmup_mode": warmup_mode,
        "warmup_initial_lr": warmup_initial_lr,
        "lr_warmup_epochs": lr_warmup_epochs,
        "initial_lr": initial_lr,
        "lr_mode": lr_mode,
        "cosine_final_lr_ratio": cosine_final_lr_ratio,
        "optimizer": optimizer,
        "optimizer_params": optimizer_params,
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": epochs,
        "mixed_precision": True,
        "loss": loss_function,
        "valid_metrics_list": metrics,
        "metric_to_watch": 'mAP@0.50:0.95'
    }

    model = models.get(
        model_name,
        num_classes=len(classes),
        pretrained_weights="coco"
    )

    trainer = Trainer(
        experiment_name=model_name,
        ckpt_root_dir=checkpoints_dir
    )

    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader
    )
