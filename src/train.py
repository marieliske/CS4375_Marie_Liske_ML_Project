# Main training script - run this to train the model 
import argparse
import base64
import csv
import os
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn as nn
from torchvision import transforms

from data import get_dataloaders
from dnn_numpy import CaptioningModel

# Edit DEFAULT_TRAINING_CONFIG below to change hyperparams
DEFAULT_TRAINING_CONFIG = {
    "captions_file": "data/captions.txt",
    "images_root": "data/Images",
    "learning_rate": 0.001,
    "epochs": 15,
    "batch_size": 32,
    "embed_dim": 256,
    "hidden_dim": 1024,
    "encoder_dim": 512,
    "min_word_freq": 5,
    "train_size": 6000,
    "val_size": 1000,
    "test_size": 1000,
    "seed": 42,
    "num_workers": 0,
    "experiment_name": "experiment_6",
    "results_dir": "results",
    "freeze_encoder": True,
    "log_every": 50,
    "beam_size": 3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Logging functions - log results in both csv and markdown formats, and also save sample captions and curves for each experiment for report
def write_log_md(path, experiment_name, params, results, caption_examples=None):
    file_path = os.path.abspath(path)
    file_exists = os.path.exists(file_path)
    experiment_number = 1

    if file_exists:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            experiment_numbers = re.findall(r"^Experiment Number:\s*(\d+)\s*$", content, flags=re.MULTILINE)
            if experiment_numbers:
                experiment_number = max(int(n) for n in experiment_numbers) + 1

    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(file_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("# Experiment Log\n\n")
        f.write(f"Experiment Number: {experiment_number}\n")
        f.write("Parameters Chosen:\n")
        f.write(f"- experiment_name: {experiment_name}\n")
        f.write(f"- learning_rate: {params['learning_rate']}\n")
        f.write(f"- batch_size: {params['batch_size']}\n")
        f.write(f"- epochs: {params['epochs']}\n")
        f.write(f"- embed_dim: {params['embed_dim']}\n")
        f.write(f"- hidden_dim: {params['hidden_dim']}\n")
        f.write(f"- min_word_freq: {params['min_word_freq']}\n")
        f.write(f"- freeze_encoder: {params['freeze_encoder']}\n")
        f.write(f"- seed: {params['seed']}\n")
        f.write("Results:\n")
        f.write(f"- timestamp: {timestamp}\n")
        f.write(
            f"- split: train:{params['train_size']} val:{params['val_size']} test:{params['test_size']}\n"
        )
        f.write(f"- best_val_loss: {results['best_val_loss']:.4f}\n")
        f.write(f"- best_val_bleu1: {results['best_val_bleu1']:.4f}\n")
        f.write(f"- test_loss: {results['test_loss']:.4f}\n")
        f.write(f"- test_bleu1: {results['test_bleu1']:.4f}\n")
        if caption_examples:
            f.write("Sample Test Captions:\n")
            for i, example in enumerate(caption_examples, start=1):
                f.write(f"- Example {i} predicted: {example['predicted']}\n")
                f.write(f"  Example {i} reference: {example['reference']}\n")
        f.write("------------------------------------------------------------\n\n")


def write_history_csv(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_bleu1"])
        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["val_loss"][i],
                    history["val_bleu1"][i],
                ]
            )


def save_caption_html(examples, out_path, experiment_name):

    def img_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    rows = ""
    for i, ex in enumerate(examples, start=1):
        try:
            b64 = img_to_b64(ex["image_path"])
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="max-width:300px;max-height:250px;">'
        except Exception:
            img_tag = f'<p><em>(image not found: {ex["image_path"]})</em></p>'
        rows += f"""
        <tr>
          <td style="text-align:center;padding:4px;">{i}</td>
          <td style="padding:4px;">{img_tag}</td>
          <td style="padding:4px;"><strong>Predicted:</strong><br>{ex['predicted']}<br>
              <strong>Reference:</strong><br><em>{ex['reference']}</em></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
        <html><head><meta charset="utf-8">
        <title>{experiment_name} – Sample Captions</title>
        <style>body{{font-family:sans-serif;font-size:28px;color:#000;margin:12px;max-width:900px;}} table{{border-collapse:collapse;}} td{{vertical-align:top;}}</style>
        </head><body>
        <h2>{experiment_name} – Sample Test Captions</h2>
        <table>{rows}
        </table></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def save_curves(history, out_path):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Captioning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_bleu1"], label="Validation BLEU-1")
    plt.title("Caption Quality")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU-1")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_log_csv(path, experiment_name, params, results):
    file_exists = os.path.exists(path)
    experiment_number = 1

    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                experiment_number = len(rows)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "experiment_number",
                    "timestamp",
                    "experiment_name",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "embed_dim",
                    "hidden_dim",
                    "min_word_freq",
                    "freeze_encoder",
                    "seed",
                    "train_size",
                    "val_size",
                    "test_size",
                    "best_val_loss",
                    "best_val_bleu1",
                    "test_loss",
                    "test_bleu1",
                ]
            )

        writer.writerow(
            [
                experiment_number,
                datetime.now().isoformat(timespec="seconds"),
                experiment_name,
                params["learning_rate"],
                params["batch_size"],
                params["epochs"],
                params["embed_dim"],
                params["hidden_dim"],
                params["min_word_freq"],
                params["freeze_encoder"],
                params["seed"],
                params["train_size"],
                params["val_size"],
                params["test_size"],
                results["best_val_loss"],
                results["best_val_bleu1"],
                results["test_loss"],
                results["test_bleu1"],
            ]
        )


@torch.no_grad()
def build_backbone_cache(encoder, datasets, device, build_batch_size=128, cache_path=None):
    # Precompute resnet features for every image and save to disk
    # Since we freeze the encoder the features never change between epochs, so no point recomputing
    from PIL import Image as PILImage

    if cache_path and os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=True)
        needed = {str(s.image_path) for ds in datasets for s in ds.samples}
        if needed.issubset(cache.keys()):
            print(f"  Loading backbone feature cache from {cache_path}")
            return cache
        # cache is outdated, need to rebuild
        print(f"  Cache at {cache_path} is stale (missing {len(needed - cache.keys())} images), rebuilding...")

    encoder.eval()
    transform = datasets[0].image_transform

    seen = set()
    unique_paths = []
    for dataset in datasets:
        for sample in dataset.samples:
            key = str(sample.image_path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(key)

    print(f"  Building backbone feature cache for {len(unique_paths)} unique images...")
    cache = {}
    for start in range(0, len(unique_paths), build_batch_size):
        batch_paths = unique_paths[start : start + build_batch_size]
        imgs = []
        for p in batch_paths:
            img = PILImage.open(p).convert("RGB")
            imgs.append(transform(img))
        imgs_t = torch.stack(imgs).to(device)
        raw = encoder.extract_raw(imgs_t)
        for j, p in enumerate(batch_paths):
            cache[p] = raw[j].cpu()

    print(f"  Cache built: {len(cache)} entries.")
    if cache_path:
        torch.save(cache, cache_path)
        print(f"  Cache saved to {cache_path}")
    return cache


def train_one_epoch(model, loader, loss_fn, opt, device, pad_idx, epoch_idx, total_epochs, log_every=20, raw_cache=None):
    model.train()
    running_loss = 0.0
    running_tokens = 0
    total_batches = len(loader)    
    for batch_idx, (images, captions_in, captions_out, _, paths) in enumerate(loader, start=1):
        captions_in = captions_in.to(device)
        captions_out = captions_out.to(device)

        if raw_cache is not None:
            raw = torch.stack([raw_cache[p] for p in paths]).to(device)
            logits = model(None, captions_in, raw_features=raw)
        else:
            logits = model(images.to(device), captions_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), captions_out.reshape(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # clip grads so training doesnt explode
        opt.step()

        non_pad = (captions_out != pad_idx).sum().item()
        running_loss += loss.item() * max(non_pad, 1)
        running_tokens += non_pad

        if batch_idx % log_every == 0 or batch_idx == total_batches:
            avg_loss = running_loss / max(running_tokens, 1)
            print(
                f"  Epoch {epoch_idx:03d}/{total_epochs} | Batch {batch_idx}/{total_batches} "
                f"| train_loss={avg_loss:.4f}"
            )

    return running_loss / max(running_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device, pad_idx, raw_cache=None):
    model.eval()
    running_loss = 0.0
    running_tokens = 0

    for images, captions_in, captions_out, _, paths in loader:
        captions_in = captions_in.to(device)
        captions_out = captions_out.to(device)

        if raw_cache is not None:
            raw = torch.stack([raw_cache[p] for p in paths]).to(device)
            logits = model(None, captions_in, raw_features=raw)
        else:
            logits = model(images.to(device), captions_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), captions_out.reshape(-1))

        non_pad = (captions_out != pad_idx).sum().item()
        running_loss += loss.item() * max(non_pad, 1)
        running_tokens += non_pad

    return running_loss / max(running_tokens, 1)


@torch.no_grad()
def get_bleu1(model, loader, vocab, device, max_batches=20, raw_cache=None, beam_size=1):
    model.eval()

    # Collect all 5 reference captions per unique image before scoring.
    # flickr8k has 5 captions per image so you need to group them before calling corpus_bleu
    img_refs = {}   # path -> list of token lists
    img_hyps = {}   # path -> hypothesis token list (computed once per image)

    for batch_idx, (images, _, captions_out, lengths, paths) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        for i in range(images.size(0)):
            path = paths[i]

            ref_ids = captions_out[i, : lengths[i]].tolist()
            ref_tokens = []
            for idx in ref_ids:
                token = vocab.itos[idx]
                if token in {"<start>", "<pad>"}:
                    continue
                if token == "<end>":
                    break
                ref_tokens.append(token)

            if ref_tokens:
                img_refs.setdefault(path, []).append(ref_tokens)

            if path not in img_hyps:
                raw_feature = raw_cache[path].to(device) if raw_cache is not None else None
                generated_ids = model.generate(
                    images[i],
                    start_idx=vocab.start_idx,
                    end_idx=vocab.end_idx,
                    max_len=25,
                    raw_feature=raw_feature,
                    beam_size=beam_size,
                )
                gen_tokens = []
                for idx in generated_ids:
                    token = vocab.itos[idx]
                    if token in {"<start>", "<pad>"}:
                        continue
                    if token == "<end>":
                        break
                    gen_tokens.append(token)
                img_hyps[path] = gen_tokens if gen_tokens else ["<unk>"]

    references = [img_refs[p] for p in img_hyps if p in img_refs]
    hypotheses = [img_hyps[p] for p in img_hyps if p in img_refs]

    if not references:
        return 0.0
    return float(corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0)))


@torch.no_grad()
def sample_captions(model, loader, vocab, device, max_examples=5, raw_cache=None, beam_size=1):
    model.eval()
    examples = []
    seen_paths = set()

    for images, _, captions_out, lengths, paths in loader:
        images = images.to(device)

        for i in range(images.size(0)):
            # each image has 5 captions so skip dupes
            if paths[i] in seen_paths:
                continue
            seen_paths.add(paths[i])
            raw_feature = raw_cache[paths[i]].to(device) if raw_cache is not None else None
            generated_ids = model.generate(
                images[i],
                start_idx=vocab.start_idx,
                end_idx=vocab.end_idx,
                max_len=25,
                raw_feature=raw_feature,
                beam_size=beam_size,
            )

            pred_tokens = []
            for idx in generated_ids:
                token = vocab.itos[idx]
                if token in {"<start>", "<pad>"}:
                    continue
                if token == "<end>":
                    break
                pred_tokens.append(token)

            ref_ids = captions_out[i, : lengths[i]].tolist()
            ref_tokens = []
            for idx in ref_ids:
                token = vocab.itos[idx]
                if token in {"<start>", "<pad>"}:
                    continue
                if token == "<end>":
                    break
                ref_tokens.append(token)

            examples.append(
                {
                    "image_path": paths[i],
                    "predicted": " ".join(pred_tokens) if pred_tokens else "<unk>",
                    "reference": " ".join(ref_tokens),
                }
            )

            if len(examples) >= max_examples:
                return examples

    return examples


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM image captioning model on Flickr8k")
    # Made it so you can pass args from command line but its easier to just edit DEFAULT_TRAINING_CONFIG above
    parser.add_argument(
        "--captions_file",
        type=str,
        default=DEFAULT_TRAINING_CONFIG["captions_file"],
        help="Path to caption file",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=DEFAULT_TRAINING_CONFIG["images_root"],
        help="Path to image folder",
    )
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_TRAINING_CONFIG["learning_rate"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAINING_CONFIG["batch_size"])
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_TRAINING_CONFIG["embed_dim"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_TRAINING_CONFIG["hidden_dim"])
    parser.add_argument("--encoder_dim", type=int, default=DEFAULT_TRAINING_CONFIG["encoder_dim"])
    parser.add_argument("--min_word_freq", type=int, default=DEFAULT_TRAINING_CONFIG["min_word_freq"])
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAINING_CONFIG["train_size"])
    parser.add_argument("--val_size", type=int, default=DEFAULT_TRAINING_CONFIG["val_size"])
    parser.add_argument("--test_size", type=int, default=DEFAULT_TRAINING_CONFIG["test_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_TRAINING_CONFIG["num_workers"])
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_TRAINING_CONFIG["experiment_name"])
    parser.add_argument("--results_dir", type=str, default=DEFAULT_TRAINING_CONFIG["results_dir"])
    parser.add_argument("--log_every", type=int, default=DEFAULT_TRAINING_CONFIG["log_every"])
    parser.add_argument("--beam_size", type=int, default=DEFAULT_TRAINING_CONFIG["beam_size"],
                        help="Beam width for caption generation (1=greedy, 3+ for beam search)")
    parser.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAINING_CONFIG["freeze_encoder"],
        help="Freeze pretrained CNN backbone",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Torchvision normalization values are the imagenet mean/std
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, val_loader, test_loader, vocab, train_dataset, val_dataset, test_dataset = get_dataloaders(
        captions_file=args.captions_file,
        images_root=args.images_root,
        image_transform=image_transform,
        batch_size=args.batch_size,
        min_word_freq=args.min_word_freq,
        split_seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
    )

    model = CaptioningModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_dim=args.encoder_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    raw_cache = None
    if args.freeze_encoder:
        cache_path = os.path.join(args.images_root, ".backbone_cache.pt")
        raw_cache = build_backbone_cache(
            model.encoder, [train_dataset, val_dataset, test_dataset], device,
            cache_path=cache_path,
        )

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_val_bleu1 = 0.0
    best_ckpt_path = os.path.join(args.results_dir, f"{args.experiment_name}_best.pt")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_bleu1": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            opt=opt,
            device=device,
            pad_idx=vocab.pad_idx,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            log_every=max(1, args.log_every),
            raw_cache=raw_cache,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            pad_idx=vocab.pad_idx,
            raw_cache=raw_cache,
        )
        val_bleu1 = get_bleu1(model=model, loader=val_loader, vocab=vocab, device=device, raw_cache=raw_cache, beam_size=args.beam_size)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bleu1={val_bleu1:.4f} "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu1"].append(val_bleu1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu1 = max(best_val_bleu1, val_bleu1)
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    # Load the best checkpoint (lowest val loss) for final test evaluation.
    # learned that the last epoch isnt always the best
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded best model from {best_ckpt_path}")

    test_loss = evaluate_loss(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        pad_idx=vocab.pad_idx,
        raw_cache=raw_cache,
    )
    test_bleu1 = get_bleu1(model=model, loader=test_loader, vocab=vocab, device=device, raw_cache=raw_cache, beam_size=args.beam_size)
    caption_examples = sample_captions(
        model=model,
        loader=test_loader,
        vocab=vocab,
        device=device,
        max_examples=5,
        raw_cache=raw_cache,
        beam_size=args.beam_size,
    )

    log_path = os.path.join(args.results_dir, "experiments_log.md")
    csv_log_path = os.path.join(args.results_dir, "experiments_log.csv")
    history_csv_path = os.path.join(args.results_dir, f"{args.experiment_name}_history.csv")
    curve_path = os.path.join(args.results_dir, f"{args.experiment_name}_curves.png")

    write_history_csv(history, history_csv_path)
    save_curves(history, curve_path)

    html_path = os.path.join(args.results_dir, f"{args.experiment_name}_captions.html")
    if caption_examples:
        save_caption_html(caption_examples, html_path, args.experiment_name)

    run_params = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "min_word_freq": args.min_word_freq,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "freeze_encoder": args.freeze_encoder,
    }
    run_results = {
        "best_val_loss": best_val_loss,
        "best_val_bleu1": best_val_bleu1,
        "test_loss": test_loss,
        "test_bleu1": test_bleu1,
    }

    write_log_csv(
        path=csv_log_path,
        experiment_name=args.experiment_name,
        params=run_params,
        results=run_results,
    )

    write_log_md(
        path=log_path,
        experiment_name=args.experiment_name,
        params=run_params,
        results=run_results,
        caption_examples=caption_examples,
    )

    print("\nFinal Summary")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Best Val BLEU1: {best_val_bleu1:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test BLEU1    : {test_bleu1:.4f}")
    print(f"History CSV saved: {history_csv_path}")
    print(f"Curves PNG saved: {curve_path}")
    if caption_examples:
        print(f"Caption HTML saved: {html_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"CSV log updated: {csv_log_path}")
    print(f"Experiment log updated: {log_path}")


if __name__ == "__main__":
    main()
