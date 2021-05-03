# trash-detection
My **training** and **evaluating** pipeline to extract *trash objects* from images

In this repo we are going to work on the [TACO dataset](http://tacodataset.org/)

I used as a training framework [Detectron2](https://github.com/facebookresearch/detectron2)

## Deployment
You can find this model deployed in my personal website: [Personal Website](https://personalwebsitemo5.vercel.app/) with TorchServe [TSS](https://github.com/Mo5mami/TSS)

If you face some **issues** with inference, The model server is hardware demanding and I can only work with free tier / limited student credits.


## Training notebook:
### Train pipeline 1
- Training on 2 parts
- Validation strategy : Stratfied group kfold
- AP@50 : 27.257
 
[Part 1](https://www.kaggle.com/mo5mami/trash-detection-project-with-taco-dataset)
[Part 2](https://www.kaggle.com/mo5mami/trash-detection-project-with-taco-dataset-part-2)

### Train pipeline 2
- Training on 2 parts
- Validation strategy : Group kfold
- Heavy augs
- AP@50 : 32.242
- Notebook not so clean but decided to put it to show different pipelines
- Although this pipeline score a lot, The other one seem to generalize better (Augs are more sensible)
- Group kfold is under-representing smaller classes

[Notebook](https://www.kaggle.com/mo5mami/trash-detection-project-with-heavier-augs)



## Our goals:
- The yaml config files **experiment.yaml** and **detectron_config.yaml** will help configurate the whole project
- **Train** an object detection model using TACO dataset
- **Evaluate** an object detection model on the train and valiation dataset


## Project structure

| File/Folder      | Description |
| ----------- | ----------- |
| configs      | The project config        |
| configs/experiment.yaml      | The general config        |
| configs/detectron_config.yaml      | Detectron2 framework config        |
| data      | The dataset repository        |
| notebooks   | The notebooks I used for training        |
| src   | The main module that is used for the training and evaluation         |
| train.py   | Script to train the model       |
| eval.py   | Script to evaluate a model        |
| requirements.txt   | Python requirements        |
| install_req   | Script to install python environment requirements        |


## Main module structure (src)

| File/Folder      | Description |
| ----------- | ----------- |
| augment.py      | The file containing the training and validation preprocess and augmentation (Albumentations)    |
| configs.py      | Main functions that are related to the config (Loading , saving , ...)        |
| evaluator.py      | The main function that help evaluate the model        |
| mapper.py      | An extension of Detectron2 DatasetMapper in order to be able to use albumentations        |
| preprocess.py   | Fix annotation problems and remove negative bboxes        |
| split.py   | Define train/val split strategy (Validation Strategy)         |
| trainer.py   | An extension of Detectron2 DefaultTrainer to add hooks (Custom checkpoints, ...)       |
| utils.py   | Util functions        |

## How to Prepare the repository:

1. Create a python environment and install dependencies

`virtualenv env`

`source env/bin/activate`

`./install_req`

2. Download the dataset and placed in data

you can find the dataset in:
- [TACO official website](http://tacodataset.org/)
- [Kaggle](https://www.kaggle.com/kneroma/tacotrashdataset)

I personally worked with the Kaggle dataset to be able to use kaggle API / train in kaggle notebooks

![image](https://user-images.githubusercontent.com/48622965/116788852-dda68100-aaa3-11eb-92d8-05ecfdb2e177.png)

## How to train the model:
`python train.py`

## How to evaluate the model:
`python eval.py` or
`python eval.py --model_path path`


## Notes
1. You can change the experiment.yaml to suit your need (I will put my best config).
2. You can change the detectron_config.yaml, but beware, some parameters in experiment.yaml overwrite some parameters in detectron_config.yaml. Read both of them carefully.
3. train.py doesn't take arguments
4. eval.py take optional argument model_path. If not specified, model_path will default to models/best_model.pth
5. Validation strategy used in the training and evaluation : Stratified group kfold:
- Each set contains approximately the same percentage of samples of each target class as the complete set.
- The same group is not represented in both testing and training sets.
6. Thank you for your patience

## Citations

**TACO dataset**
```
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
```
**Detectron2**

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```




