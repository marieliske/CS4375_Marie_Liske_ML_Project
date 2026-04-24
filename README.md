# Image Captioning with CNN + LSTM (Flickr8k)

This project now follows the topic from your status report:

- Encoder-decoder captioning architecture
- Pretrained CNN (ResNet-50) for image feature extraction
- LSTM decoder trained to generate captions word-by-word
- Training/evaluation in Python with PyTorch

## Team

- Marie Liske (MXL210044)

## Project Structure

- `src/data.py`: Flickr8k caption parsing, train/val/test split, vocabulary, dataloaders.
- `src/dnn_numpy.py`: model definitions (`EncoderCNN`, manual LSTM decoder, `CaptioningModel`).
- `src/train.py`: training loop, validation loss + BLEU-1, metrics/history/checkpoint saving.
- `src/run_experiments.py`: runs multiple captioning experiments and appends to log.
- `results/`: generated experiment outputs.
- `report/ieee_report_starter.md`: report starter sections.

## Dataset

Target dataset: Flickr8k.

- 8,000 images
- 5 captions per image (40,000 captions total)
- Default split in this project: 6,000 train / 1,000 val / 1,000 test by image

Expected inputs:

- Caption file (`Flickr8k.token.txt` style `image.jpg#0<TAB>caption` or CSV `image.jpg,caption`)
- Image directory containing Flickr8k image files

## Environment Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run One Training Experiment

```bash
python src/train.py \
  --captions_file path/to/Flickr8k.token.txt \
  --images_root path/to/Flickr8k_Images \
  --experiment_name flickr8k_baseline \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --embed_dim 256 \
  --hidden_dim 512 \
  --min_word_freq 5 \
  --freeze_encoder
```

## Run Multiple Experiments

```bash
python src/run_experiments.py path/to/Flickr8k.token.txt path/to/Flickr8k_Images
```

## Outputs Generated

For each experiment, files are saved in `results/`:

- `<experiment_name>_metrics.json`
- `<experiment_name>_history.csv`
- `<experiment_name>_curves.png`
- `<experiment_name>_best.pt`

Also maintained:

- `results/experiments_log.csv`

## Metrics

- Training loss
- Validation loss
- Validation BLEU-1
- Test loss
- Test BLEU-1

## Requirement Mapping to Report

- Topic: image and video captioning with DNNs (implemented image captioning path)
- CNN for visual feature extraction: pretrained ResNet-50 encoder
- Sequence generation model: LSTM decoder equations are implemented directly in project code (manual input/forget/cell/output gates)
- Libraries: PyTorch/torchvision, NLTK, Matplotlib, NumPy/Pandas

## What Is Coded by You

- Core algorithm coded in this repo:
  - LSTM decoder gate equations and recurrent rollout in `src/dnn_numpy.py`
  - Caption generation loop (token-by-token inference) in `src/dnn_numpy.py`
  - End-to-end training loop, loss computation, and evaluation flow in `src/train.py`
- Library usage (allowed by your project rules):
  - Dataset loading/tokenization/transforms
  - Pretrained CNN feature extractor (transfer learning)
  - Metrics and plotting utilities

## Dataset Hosting (for TA Reproducibility)

Choose one public hosting option and keep links in your report and README.

Option A: Kaggle dataset link (fastest)

1. Upload your prepared Flickr8k folder structure to a Kaggle Dataset.
2. Mark it public.
3. Add the dataset URL to README and report.
4. In instructions, specify expected paths for `--captions_file` and `--images_root` after download.

Option B: GitHub Releases (good for moderate-size zip)

1. Create a zip of dataset files (or metadata + split files if large).
2. Create a release in your project repo.
3. Attach the zip as a release asset.
4. Add release URL and a one-command download step in README.

Option C: AWS S3 public bucket (recommended for large files)

1. Create bucket: `aws s3 mb s3://<your-bucket-name>`
2. Upload files: `aws s3 cp <local_dataset_folder> s3://<your-bucket-name>/flickr8k/ --recursive`
3. Make objects public (or use presigned links):
   - `aws s3api put-bucket-policy --bucket <your-bucket-name> --policy file://bucket-policy.json`
4. Add direct links plus download command in README:
   - `aws s3 sync s3://<your-bucket-name>/flickr8k/ data/flickr8k/`

Minimum deliverable requirement for dataset hosting:

1. Public URL
2. Exact folder/file layout expected by training command
3. One copy-paste download command
4. A checksum or file count note so TA can verify download integrity

## Notes

- This code implements image captioning on Flickr8k.
- Video captioning can be added later by replacing single-image features with per-frame feature sequences and temporal aggregation.
