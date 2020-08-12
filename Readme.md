# Human-Protein-Classifier-using-Pytorch

## Dataset:
Dataset contains 512x512 resolution images of 3 channels (RGB). Images are in PNG format. Train set contains - 19236 images Test set contains - 8243 images (Public leaderboard results are from just 40%(3297 images) of this and rest 60% will be used for private leaderboard)

## Model Built:

1. In this I used PytorchTransforms

```
train_tfms = T.Compose([
    T.RandomCrop(512, padding=8, padding_mode='reflect'),
#     T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(), 
#     T.Normalize(*imagenet_stats,inplace=True), 
    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
#     T.Resize(256), 
    T.ToTensor(), 
#     T.Normalize(*imagenet_stats)
])

```

2. batch size of 64, Resenet 18(Pre-Trained Model), One Cycle Learning Rate and I ran the model with

```
num_epochs = 6
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

```

### Predicted Image

![image](https://user-images.githubusercontent.com/61023747/90020022-e0223a00-dccc-11ea-9a5e-48249d6ad41c.png)


## Achievement

**Secured 218th rank among 894 i.e.,Top 25%** in [Kaggle In-Class Competition](https://www.kaggle.com/c/jovian-pytorch-z2g/data) which was conducted by [Jovian.ml](https://www.jovian.ml/)

 "Might be not a good rank but I gave my best, my first Step towards the Data Science"

