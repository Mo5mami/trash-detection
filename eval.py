from detectron2.utils.logger import setup_logger


import pandas as pd
import argparse

from src.trainer import train
from src.evaluator import evaluate
from src.utils import load_annotations
from src.preprocess import fix_annotations , choose_category
from src.split import kfold_split

def eval_function(path):
    fix_annotations()
    annot = load_annotations(preprocessed=True)
    annot_df=pd.DataFrame(annot["annotations"])
    images_df=pd.DataFrame(annot["images"])
    annot_df , annot = choose_category(annot_df , annot)
    annot_df = kfold_split(annot_df)
    #print(annot_df.head())
    evaluate(annot_df , images_df , annot , path)




if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description='Run evaluation on train and validation dataset')
    parser.add_argument('--model_path', required=False, default="models/best_model.pth", help="Path to the weights")
    args = parser.parse_args()
    eval_function(args.model_path)
    #from src.configs import load_general_config , load_detectron_config
    #cfg = load_detectron_config()
    #print(cfg)