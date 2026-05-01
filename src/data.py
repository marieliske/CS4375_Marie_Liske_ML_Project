import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


# Special tokens the vocab needs
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"


# Regex splits on anything thats not a letter/number/apostrophe
def split_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# builds a vocab from the training captions - maps words to ints and back
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
        word_counts = Counter()
        for cap in captions:
            word_counts.update(split_words(cap))

        # only keep words that appear at least min_freq times
        for token, count in word_counts.items():
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, caption):
        words = split_words(caption)
        ids = [self.stoi[START_TOKEN]]
        ids.extend(self.stoi.get(word, self.stoi[UNK_TOKEN]) for word in words)
        ids.append(self.stoi[END_TOKEN])
        return ids

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


def load_samples(captions_file, images_root):
    captions_path = Path(captions_file)
    images_root = Path(images_root)
    samples = []

    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            img_name = None
            caption = None

            if "\t" in line and "#" in line.split("\t", maxsplit=1)[0]:
                left, caption = line.split("\t", maxsplit=1)
                img_name = left.split("#", maxsplit=1)[0]
            elif "," in line:
                img_name, caption = line.split(",", maxsplit=1)

            if img_name is None or caption is None:
                continue

            img_path = images_root / img_name.strip()
            if img_path.exists():
                samples.append(CaptionSample(image_path=img_path, caption=caption.strip()))

    if not samples:
        raise ValueError("No samples loaded. Check captions_file format and images_root path.")

    return samples


# Splits the dataset by image so the same image doesn't appear in both train and val
def split_samples(samples, train_size=6000, val_size=1000, test_size=1000, seed=42):
    img_to_caps = {}
    for row in samples:
        img_to_caps.setdefault(row.image_path, []).append(row.caption)

    image_paths = list(img_to_caps.keys())
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    n_images = len(image_paths)
    if n_images < (train_size + val_size + test_size):
        raise ValueError(
            "Not enough unique images for requested split sizes. "
            f"Found {n_images}, need at least {train_size + val_size + test_size}."
        )

    train_imgs = set(image_paths[:train_size])
    val_imgs = set(image_paths[train_size : train_size + val_size])
    test_imgs = set(image_paths[train_size + val_size : train_size + val_size + test_size])

    train_rows = [CaptionSample(image_path=img_path, caption=cap) for img_path in train_imgs for cap in img_to_caps[img_path]]
    val_rows = [CaptionSample(image_path=img_path, caption=cap) for img_path in val_imgs for cap in img_to_caps[img_path]]
    test_rows = [CaptionSample(image_path=img_path, caption=cap) for img_path in test_imgs for cap in img_to_caps[img_path]]

    return train_rows, val_rows, test_rows


class FlickrCaptionDataset(Dataset):
    def __init__(self, samples, vocab, image_transform):
        self.samples = samples
        self.vocab = vocab
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        image = Image.open(row.image_path).convert("RGB")
        image = self.image_transform(image)

        ids = self.vocab.encode(row.caption)
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)

        return image, inputs, targets, str(row.image_path)


def make_batch(batch, pad_idx):
    imgs, in_seqs, out_seqs, paths = zip(*batch)

    images = torch.stack(imgs, dim=0)
    max_len = max(seq.shape[0] for seq in in_seqs)

    input_batch = torch.full((len(in_seqs), max_len), pad_idx, dtype=torch.long)
    target_batch = torch.full((len(out_seqs), max_len), pad_idx, dtype=torch.long)
    lengths = []

    for i, (inp, tgt) in enumerate(zip(in_seqs, out_seqs)):
        seq_len = inp.shape[0]
        input_batch[i, :seq_len] = inp
        target_batch[i, :seq_len] = tgt
        lengths.append(seq_len)

    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, input_batch, target_batch, lengths, list(paths)


def get_dataloaders(captions_file, images_root, image_transform, batch_size=32, min_word_freq=5, split_seed=42, train_size=6000, val_size=1000, test_size=1000, num_workers=0):
    samples = load_samples(captions_file=captions_file, images_root=images_root)
    train_samples, val_samples, test_samples = split_samples(samples=samples, train_size=train_size, val_size=val_size, test_size=test_size, seed=split_seed)
    vocab = Vocabulary(min_freq=min_word_freq)
    vocab.build([sample.caption for sample in train_samples])

    train_dataset = FlickrCaptionDataset(train_samples, vocab=vocab, image_transform=image_transform)
    val_dataset = FlickrCaptionDataset(val_samples, vocab=vocab, image_transform=image_transform)
    test_dataset = FlickrCaptionDataset(test_samples, vocab=vocab, image_transform=image_transform)

    collate_fn = lambda batch: make_batch(batch, pad_idx=vocab.pad_idx)

    # Shuffle only train, not val/test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab, train_dataset, val_dataset, test_dataset
