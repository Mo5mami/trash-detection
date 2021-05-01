from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader , build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import os

from .mapper import PersonalMapper
from .configs import load_general_config , load_detectron_config , inject_config , dump_dict
from .configs import MODELS_PATH, LOGS_PATH
from .utils import register_dataset , seed_all

@inject_config
def evaluate(config,annot_df , images_df , annot , path):
    """
    train function that help train on the dataset and validate on a certain fold
    """
    seed_all()
    fold = config.general["fold"]
    register_dataset(annot_df , images_df , annot)
    cfg = load_detectron_config()
    metrics={}
    cfg.MODEL.WEIGHTS = path
    model = build_model(cfg)
    m=DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = COCOEvaluator(f"my_dataset_test_{fold}", ("bbox",), False, output_dir=LOGS_PATH)
    #val_loader = build_detection_test_loader(cfg, f"my_dataset_test_{fold}")
    loader = build_detection_test_loader( cfg,f"my_dataset_test_{fold}",mapper=PersonalMapper(cfg,is_train=False,augmentations=[]))
    val_metric=inference_on_dataset(model, loader, evaluator)
    metrics["validation_metric"]=val_metric

    evaluator = COCOEvaluator(f"my_dataset_train_{fold}", ("bbox",), False, output_dir=LOGS_PATH)
    #val_loader = build_detection_test_loader(cfg, f"my_dataset_train_{fold}")
    loader = build_detection_test_loader( cfg,f"my_dataset_train_{fold}",mapper=PersonalMapper(cfg,is_train=False,augmentations=[]))
    train_metric=inference_on_dataset(model, loader, evaluator)
    metrics["train_metric"]=train_metric
    dump_dict(metrics,os.path.join(LOGS_PATH,"metrics.yaml"))