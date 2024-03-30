# SM-GAN

This is the code for the paper:
SM-GAN: Single-stage and Multi-object Text Guided Image Editing


## Installation

Clone this repo and go to the cloned directory.

Please create a environment using python 3.7 and install dependencies by
```bash
pip install -r requirements.txt
```

To reproduce the results reported in the paper, you would need an V100 GPU.

## Download datasets and pretrained model
The original Clevr dataset we used is from this [external website](https://github.com/google/tirg). The original Abstract Scene we used is from this [external website](https://github.com/Maluuba/GeNeVA_datasets/).



## Training

New models can be trained with the following commands.

1. Prepare dataset. Follow the structure of the provided
datasets, which means you need paired data (input image, input text, output image)

2. Train.

```bash
# Pretraining
bash run_pretrain.sh

# Training
bash run_train.sh
```


## Testing

```bash
bash run_test.sh
```



## Code Structure

- `dataset/`: defines the class for loading the dataset. Place the dataset in this folder.

