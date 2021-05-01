DATASET_PATH = "data/trash-taco-dataset"
LOGS_PATH = "logs"
MODELS_PATH = "models"
CONFIG_PATH = "configs"

from detectron2.config import get_cfg
from detectron2 import model_zoo

from yacs.config import CfgNode as CN
from functools import wraps
import os
import yaml

def load_general_config(path = os.path.join(CONFIG_PATH , "experiment.yaml") ):
    _C = CN()
    _C.general=CN()
    _C.general.seed = 42
    _C.general.n_folds = 5
    _C.general.fold = 0
    _C.general.tool = "detectron2"
    _C.general.experiment_id = "26-04-2021"
    _C.general.category = "super_category"
    _C.general.augmentations = True
    _C.general.TTA = False

    _C.preprocess=CN()
    _C.preprocess.height = 1500
    _C.preprocess.width = 1500
    _C.preprocess.longest_max_size = 1500
    _C.preprocess.smallest_max_size = 1000

    _C.model=CN()
    _C.model.base_lr = 0.001
    _C.model.num_classes = 29 #29 if super category 60 if normal category 
    _C.model.model_name = "faster_rcnn_R_101_FPN_3x"
    _C.model.batchsize_per_image = 1024
    #_C.model.images_per_batch = 4
    _C.model.images_per_batch = 4
    _C.model.epochs = 9
    cfg = _C
    cfg.merge_from_file(path)
    return cfg

def inject_config(funct):
    """Inject a yacs CfgNode object in a function as first arg."""
    @wraps(funct)
    def function_wrapper(*args,**kwargs):
        return funct(load_general_config(),*args,**kwargs)  
    return function_wrapper


@inject_config
def load_detectron_config(config , path = os.path.join(CONFIG_PATH , "detectron_config.yaml") ):
    cfg = get_cfg()
    cfg.OUTPUT_DIR_BEST = LOGS_PATH
    cfg.merge_from_file(path)
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{config.model['model_name']}.yaml"))
    #cfg.MODEL.WEIGHTS = None
    fold = config.general["fold"]
    train_dataset_name=f"my_dataset_train_{fold}"
    test_dataset_name=f"my_dataset_test_{fold}"
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.OUTPUT_DIR_BEST = LOGS_PATH
    return cfg


def dump_cfg(config , path = "experiment.yaml"):
    """Save a yacs CfgNode object in a yaml file in path."""
    stream = open(path, 'w')
    stream.write(config.dump())
    stream.close()



def dump_dict(config,path="config.yaml"):
        stream = open(path, 'w')
        yaml.dump(config,stream)
        stream.close()