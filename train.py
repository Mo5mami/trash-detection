from detectron2.utils.logger import setup_logger


import pandas as pd

from src.trainer import train
from src.utils import load_annotations
from src.preprocess import fix_annotations , choose_category
from src.split import kfold_split

def train_function():
    fix_annotations()
    annot = load_annotations(preprocessed=True)
    annot_df=pd.DataFrame(annot["annotations"])
    images_df=pd.DataFrame(annot["images"])
    annot_df , annot = choose_category(annot_df , annot)
    annot_df = kfold_split(annot_df)
    train(annot_df , images_df , annot)




if __name__ == "__main__":
    setup_logger()
    train_function()