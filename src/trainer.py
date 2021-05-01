import detectron2
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader , build_detection_train_loader

from .mapper import PersonalMapper
from .configs import load_general_config , load_detectron_config , inject_config
from .configs import MODELS_PATH
from .utils import register_dataset , seed_all

class PersonalTrainer (detectron2.engine.defaults.DefaultTrainer):
    """
    Personal trainer based on detectron2 DefaultTrainer to add some hooks and change data loaders
    """
    
    def __init__(self, cfg , config=load_general_config() , steps_per_epoch=1100):
        self.steps_per_epoch = steps_per_epoch
        super().__init__(cfg)
        self.metric=0
        self.checkpointer.save_dir=MODELS_PATH
        

        
    def build_hooks(self):
        hooks = super().build_hooks()
        def save_best_model():
            
            metric=self.test(self.cfg, self.model)["bbox"]["AP50"]
            if(metric>self.metric):
                self.metric=metric
                self.checkpointer.save("best_model") # it will add .pth alone
                
        model_checkpointer=EvalHook(self.steps_per_epoch, save_best_model)
        hooks.insert(-1,model_checkpointer)
        return hooks
    
    @classmethod
    def build_train_loader(cls, cfg):
        
        return build_detection_train_loader(cfg,mapper=PersonalMapper(cfg,is_train=True,augmentations=[]))
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        
        return build_detection_test_loader( cfg,dataset_name,mapper=PersonalMapper(cfg,is_train=False,augmentations=[]))

    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, ("bbox",), False, output_dir=None)

 
@inject_config
def train(config,annot_df , images_df , annot):
    """
    train function that help train on the dataset and validate on a certain fold
    """
    seed_all()
    fold = config.general["fold"]
    register_dataset(annot_df , images_df , annot)
    cfg = load_detectron_config()
    steps_per_epoch = annot_df.shape[0]//config.model["images_per_batch"]
    trainer = PersonalTrainer(cfg , steps_per_epoch = steps_per_epoch ) 
    trainer.resume_or_load(resume=False)
    trainer.evaluator = COCOEvaluator(f"my_dataset_test_{fold}", ("bbox",), False, output_dir=None)
    trainer.train()