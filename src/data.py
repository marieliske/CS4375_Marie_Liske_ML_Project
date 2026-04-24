import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.stoi = {
            PAD_TOKEN: 0,
            START_TOKEN: 1,
            END_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self.itos = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

    def build(self, captions):
        counts = Counter()
        for caption in captions:
            counts.update(tokenize(caption))

        for token, count in counts.items():
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, caption):
        tokens = tokenize(caption)
        token_ids = [self.stoi[START_TOKEN]]
        token_ids.extend(self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens)
        token_ids.append(self.stoi[END_TOKEN])
        return token_ids

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            token = self.itos[idx]
            if token in (PAD_TOKEN, START_TOKEN, END_TOKEN):
                continue
            words.append(token)
        return " ".join(words)

    @property
    def pad_idx(self):
        return self.stoi[PAD_TOKEN]

    @property
    def start_idx(self):
        return self.stoi[START_TOKEN]

    @property
    def end_idx(self):
        return self.stoi[END_TOKEN]

    def __len__(self):
        return len(self.itos)


@dataclass
class CaptionSample:
    image_path: Path
    caption: str


def load_flickr8k_samples(captions_file, images_root):
    """
    Supports two common formats:
    - Flickr8k.token.txt style: image.jpg#0\tA caption...
    - CSV style: image.jpg,caption text
    """
    captions_path = Path(captions_file)
    images_root = Path(images_root)
    samples = []

    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            image_name = None
            caption = None

            if "\t" in line and "#" in line.split("\t", maxsplit=1)[0]:
                left, caption = line.split("\t", maxsplit=1)
                image_name = left.split("#", maxsplit=1)[0]
            elif "," in line:
                image_name, caption = line.split(",", maxsplit=1)

            if image_name is None or caption is None:
                continue

            img_path = images_root / image_name.strip()
            if img_path.exists():
                samples.append(CaptionSample(image_path=img_path, caption=caption.strip()))

    if not samples:
        raise ValueError("No samples loaded. Check captions_file format and images_root path.")

    return samples


def split_by_image(samples, train_size=6000, val_size=1000, test_size=1000, seed=42):
    image_to_captions = {}
    for sample in samples:
        image_to_captions.setdefault(sample.image_path, []).append(sample.caption)

    image_paths = list(image_to_captions.keys())
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    if len(image_paths) < (train_size + val_size + test_size):
        raise ValueError("Not enough unique images for the requested split sizes.")

    train_imgs = set(image_paths[:train_size])
    val_imgs = set(image_paths[train_size : train_size + val_size])
    test_imgs = set(image_paths[train_size + val_size : train_size + val_size + test_size])

    def flatten(img_set):
        out = []
        for image_path in img_set:
            for caption in image_to_captions[image_path]:
                out.append(CaptionSample(image_path=image_path, caption=caption))
        return out

    return flatten(train_imgs), flatten(val_imgs), flatten(test_imgs)


class FlickrCaptionDataset(Dataset):
    def __init__(self, samples, vocab, image_transform):
        self.samples = samples
        self.vocab = vocab
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.image_transform(image)

        token_ids = self.vocab.encode(sample.caption)
        inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)

        return image, inputs, targets


def collate_caption_batch(batch, pad_idx):
    images, inputs, targets = zip(*batch)

    images = torch.stack(images, dim=0)
    max_len = max(seq.shape[0] for seq in inputs)

    input_batch = torch.full((len(inputs), max_len), pad_idx, dtype=torch.long)
    target_batch = torch.full((len(targets), max_len), pad_idx, dtype=torch.long)
    lengths = []

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        seq_len = inp.shape[0]
        input_batch[i, :seq_len] = inp
        target_batch[i, :seq_len] = tgt
        lengths.append(seq_len)

    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, input_batch, target_batch, lengths


def build_dataloaders(
    captions_file,
    images_root,
    image_transform,
    batch_size=32,
    min_word_freq=5,
    split_seed=42,
    num_workers=0,
):
    samples = load_flickr8k_samples(captions_file=captions_file, images_root=images_root)
    train_samples, val_samples, test_samples = split_by_image(samples=samples, seed=split_seed)

    vocab = Vocabulary(min_freq=min_word_freq)
    vocab.build([sample.caption for sample in train_samples])

    train_dataset = FlickrCaptionDataset(train_samples, vocab=vocab, image_transform=image_transform)
    val_dataset = FlickrCaptionDataset(val_samples, vocab=vocab, image_transform=image_transform)
    test_dataset = FlickrCaptionDataset(test_samples, vocab=vocab, image_transform=image_transform)

    collate = lambda batch: collate_caption_batch(batch, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, vocab
