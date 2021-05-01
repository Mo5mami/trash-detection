from .configs import inject_config
import albumentations as A

@inject_config
def get_train_transforms(config):
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit= 0.1, 
                                     val_shift_limit=0.1, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, 
                                           contrast_limit=0.2, p=0.8),
            ],p=0.7),
            A.Rotate (limit=15, interpolation=1, border_mode=4, value=None, mask_value=None, p=0.8),
            
            
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop (config.preprocess.height, config.preprocess.width, scale=(0.8, 0.8), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=0.1),
            A.OneOf([
            A.Resize(height=config.preprocess.height, width=config.preprocess.width, p=0.2),
            A.LongestMaxSize(max_size=config.preprocess.longest_max_size, p=0.2),
            A.SmallestMaxSize(max_size=config.preprocess.smallest_max_size, p=0.2),
                
            ], p=1),
            A.CLAHE(clip_limit=[1,4],p=1),
            
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='coco',
            min_area=0.5, 
            min_visibility=0.5,
            label_fields=['category_id']
        )
    )

@inject_config
def get_valid_transforms(config):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=config.preprocess.smallest_max_size, p=1.0),
            A.CLAHE(clip_limit=[3,3],p=1),   
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='coco',
            min_area=0.5, 
            min_visibility=0.5,
            label_fields=['category_id']
        )
    )

def get_transforms(train=True):
    if (train):
        return get_train_transforms()
    return get_valid_transforms()