# SSC-GAN_repo
Pytorch implementation for 'Semi-Supervised Single-Stage Controllable GANs for Conditional Fine-Grained Image Generation'.[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Semi-Supervised_Single-Stage_Controllable_GANs_for_Conditional_Fine-Grained_Image_Generation_ICCV_2021_paper.pdf)


### SSC-GAN:Semi-Supervised Single-Stage Controllable GANs for Conditional Fine-Grained Image Generation
Authors : Tianyi Chen, Yi Liu, Yunfei Zhang, Si Wu, Yong Xu, Feng Liangbing and Hau San Wong

## Requirements
- Linux or Windows 
- Python 3.6+
- Pytorch 1.2.0+

## Getting started
### Clone the repository
```bash
git clone https://github.com/bravotty/SSC-GAN_repo
cd SSC-GAN_repo
```
### Setting up the data
**Note**: You need to download the data if you wish to train your own model.

Download the formatted CUB data from this [link](https://pan.baidu.com/s/1oEHcskg74FGZG9A2TuWavA)[BaiDuYunDisk] and its extracted code: xbq4 and extract it inside the `data` directory
```bash
cd data
unzip birds.zip
cd ..
```

### Downloading pretrained models
Pretrained generator models for CUB are available at this [link](https://pan.baidu.com/s/1Skzwv7e8IK8KaKpShQSxXQ)[BaiDuYunDisk] and its extracted code:4ko5. Download and extract them in the `models_pth` directory.

## Evaluating the model
In `cfg/eval.yml`:
- Specify the model path in `TRAIN.NET_G`.
- Specify the output directory to save the generated images in `SAVE_DIR`. 
- Run `python main.py --cfg cfg/eval.yml`

## Training your own model
In `cfg/train.yml`:
- Specify the dataset location in `DATA_DIR`.
- Specify the number of fine-grained categories that you wish for SReGAN, in `CLASSES`.
- Specify the training hyperparameters in `TRAIN`.
- Run `python main.py --cfg cfg/train.yml`

## Acknowledgement
We thank the authors of FineGAN: Unsupervised Hierarchical Disentanglement for Fine-grained Object Generation and Discovery for releasing their source code.

