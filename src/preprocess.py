import json
import os 

from .configs import DATASET_PATH,inject_config
from .utils import get_normal_categories,get_super_categories

def fix_annotations(path = os.path.join(DATASET_PATH,"data/annotations.json" ) ,
result_path = os.path.join(DATASET_PATH,"data/new_annotations.json" )):
    """
        repeated annotations idx 
        308 => 0
        4039  =>2197
    """
    annot=json.load(open(path))
    annot["annotations"][308]["id"]=0
    annot["annotations"][4039]["id"]=2197

    # Delete negative bboxes
    annot_to_delete=[]
    for idx,annotation in enumerate(annot["annotations"]):
        if (annotation["bbox"][0]<0 or annotation["bbox"][1]<0 or
            annotation["bbox"][2]<0 or annotation["bbox"][3]<0):
            annot_to_delete.append(idx)
    for pos,idx in enumerate(annot_to_delete):
        del annot["annotations"][idx-pos]

    # Save result
    json.dump(annot,open(result_path,"w"))

@inject_config
def choose_category(config , annot_df , annot):

    categories = get_normal_categories()
    super_categories = get_super_categories()
    annot_df["category"]=annot_df["category_id"].apply(lambda value : categories[value])
    annot_df["super_category"]=annot_df["category_id"].apply(lambda value : super_categories[value])
    super_category_to_index={value : key for key,value in enumerate(annot_df["super_category"].unique())}
    annot_df["super_category_id"]=annot_df["super_category"].apply(lambda value : super_category_to_index[value])
    annot_df["normal_category_id"]=annot_df["category_id"]
    annot_df["normal_category"]=annot_df["category"]
    if config.general["category"] != "normal_category":
        annot_df["category_id"]=annot_df["super_category_id"]
        annot_df["category"]=annot_df["super_category"]
        annot_cat=annot_df.groupby("category_id")[["category_id","category","super_category"]].first()
        annot_cat.columns=["id","name","supercategory"]
        annot["categories"]=annot_cat.to_dict("records")
    
    return annot_df , annot