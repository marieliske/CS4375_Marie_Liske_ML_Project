# CS 4375 Final Project - Image Captioning

Marie Liske - CS4375.004 - ML Final Project: Image Captioning Model using DNN.

## What this does

Trains an image captioning model on the Flickr8k dataset. The model takes an image and outputs a sentence describing it.

- Encoder: ResNet-50 pretrained on ImageNet (I froze the weights and just use it as a feature extractor)
- Decoder: LSTM I implemented myself by hand (the gate equations are all written out explicitly, not using nn.LSTM)
- Outputs: loss/BLEU curves, a checkpoint file, and an experiment log

## Setup

You need Python 3.10+ and a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download the dataset

The Flickr8k dataset is hosted on GitHub releases (its about 1GB):

```powershell
python src/download_data.py
```

This will take a few minutes. After it finishes you should have:

- data/captions.txt
- data/Images/

## Training

Before running, open `src/train.py` and edit `DEFAULT_TRAINING_CONFIG` at the top of the file to set your hyperparameters:

- `learning_rate`
- `epochs`
- `batch_size`
- `embed_dim`, `hidden_dim`
- `min_word_freq`
- `train_size`, `val_size`, `test_size`
- `experiment_name` (change this each run so results dont overwrite)
- `freeze_encoder`

Then run:

```powershell
python src/train.py
```

The first run will take extra time to build a feature cache for all the images (saves to `data/Images/.backbone_cache.pt`). Subsequent runs load the cache and are much faster.

## Results

After each run, results are saved to the `results/` folder:

- `<experiment_name>_history.csv` - loss/bleu per epoch
- `<experiment_name>_curves.png` - training curves plot
- `<experiment_name>_captions.html` - sample predicted captions with images
- `<experiment_name>_best.pt` - best model checkpoint

The full log across all runs is in:

- results/experiments_log.csv
- results/experiments_log.md
