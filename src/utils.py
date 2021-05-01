import torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import numpy as np
import random
import json
import os

from .configs import inject_config,DATASET_PATH


@inject_config
def seed_all(config):
    """
    seed my experiments to be able to reproduce
    """
    seed_value=config.general["seed"]
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


@inject_config
def register_dataset(config , annot_df , images_df , annot):
    """
    Register dataset (detectron2 register coco instance)
    folds included
    """
    fold = config.general["fold"]
    train_dataset_name=f"my_dataset_train_{fold}"
    test_dataset_name=f"my_dataset_test_{fold}"
    train_dataset_file=os.path.join(DATASET_PATH,f"my_dataset_train_{fold}.json")
    test_dataset_file=os.path.join(DATASET_PATH,f"my_dataset_test_{fold}.json")
    
    train_annot_df=annot_df[annot_df["folds"]!=fold]
    test_annot_df=annot_df[annot_df["folds"]==fold]
    train_annot_df=train_annot_df.drop(["normal_category","normal_category_id"],axis=1)
    test_annot_df=test_annot_df.drop(["normal_category","normal_category_id"],axis=1)

    train_images_df=images_df[images_df["id"].apply(lambda i:True if i in list(train_annot_df["image_id"].unique()) else False)]
    test_images_df=images_df[images_df["id"].apply(lambda i:True if i in list(test_annot_df["image_id"].unique()) else False)]
    
    train_annot=annot.copy()
    test_annot=annot.copy()
    
    train_annot["annotations"]=train_annot_df.reset_index(drop=True).to_dict("records")
    train_annot["images"]=train_images_df.reset_index(drop=True).to_dict("records")
    test_annot["annotations"]=test_annot_df.reset_index(drop=True).to_dict("records")
    test_annot["images"]=test_images_df.reset_index(drop=True).to_dict("records")
    
    json.dump(train_annot,open(train_dataset_file,"w"))
    json.dump(test_annot,open(test_dataset_file,"w"))
    
    if train_dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(train_dataset_name)
        MetadataCatalog.remove(train_dataset_name)
    if test_dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(test_dataset_name)
        MetadataCatalog.remove(test_dataset_name)
        
    register_coco_instances(train_dataset_name, {}, train_dataset_file, os.path.join(DATASET_PATH,"data"))
    register_coco_instances(test_dataset_name, {}, test_dataset_file, os.path.join(DATASET_PATH,"data"))


def load_annotations(preprocessed = True):
    """
    load json annotations as dict
    """
    if preprocessed:
        annot=json.load(open(os.path.join(DATASET_PATH,"data/new_annotations.json")))
    else :
        annot=json.load(open(os.path.join(DATASET_PATH,"data/annotations.json")))
    return annot


def get_normal_categories():
    """
    normal categories
    get a dict : annotation id -> annotation name
    """
    annot = load_annotations(preprocessed = True)
    categories={ annotation["id"] : annotation["name"] for annotation in annot["categories"]}
    return categories

def get_super_categories():
    """
    super categories
    get a dict : annotation id -> annotation name
    """
    annot = load_annotations(preprocessed = True)
    super_categories={ annotation["id"] : annotation["supercategory"] for annotation in annot["categories"]}
    return super_categories