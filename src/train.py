import argparse
import csv
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn as nn
from torchvision import transforms

from data import build_dataloaders
from dnn_numpy import CaptioningModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history_csv(history, path):
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


def append_experiment_log(path, experiment_name, params, best_val_loss, best_val_bleu1):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "experiment_name",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "embed_dim",
                    "hidden_dim",
                    "min_word_freq",
                    "seed",
                    "best_val_loss",
                    "best_val_bleu1",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                experiment_name,
                params["learning_rate"],
                params["batch_size"],
                params["epochs"],
                params["embed_dim"],
                params["hidden_dim"],
                params["min_word_freq"],
                params["seed"],
                best_val_loss,
                best_val_bleu1,
            ]
        )


def plot_history(history, out_path):
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


def run_epoch(model, dataloader, criterion, optimizer, device, pad_idx):
    model.train()
    running_loss = 0.0
    running_tokens = 0

    for images, captions_in, captions_out, _ in dataloader:
        images = images.to(device)
        captions_in = captions_in.to(device)
        captions_out = captions_out.to(device)

        logits = model(images, captions_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        non_pad = (captions_out != pad_idx).sum().item()
        running_loss += loss.item() * max(non_pad, 1)
        running_tokens += non_pad

    return running_loss / max(running_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device, pad_idx):
    model.eval()
    running_loss = 0.0
    running_tokens = 0

    for images, captions_in, captions_out, _ in dataloader:
        images = images.to(device)
        captions_in = captions_in.to(device)
        captions_out = captions_out.to(device)

        logits = model(images, captions_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions_out.reshape(-1))

        non_pad = (captions_out != pad_idx).sum().item()
        running_loss += loss.item() * max(non_pad, 1)
        running_tokens += non_pad

    return running_loss / max(running_tokens, 1)


@torch.no_grad()
def evaluate_bleu1(model, dataloader, vocab, device, max_batches=20):
    model.eval()
    references = []
    hypotheses = []

    for batch_idx, (images, _, captions_out, lengths) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        for i in range(images.size(0)):
            generated_ids = model.generate(
                images[i],
                start_idx=vocab.start_idx,
                end_idx=vocab.end_idx,
                max_len=25,
            )

            gen_tokens = []
            for idx in generated_ids:
                token = vocab.itos[idx]
                if token in {"<start>", "<pad>"}:
                    continue
                if token == "<end>":
                    break
                gen_tokens.append(token)

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
                references.append([ref_tokens])
                hypotheses.append(gen_tokens if gen_tokens else ["<unk>"])

    if not references:
        return 0.0
    return float(corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0)))


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM image captioning model on Flickr8k")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to Flickr8k caption file")
    parser.add_argument("--images_root", type=str, required=True, help="Path to Flickr8k image folder")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--min_word_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="flickr8k_baseline")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze pretrained CNN backbone")
    parser.add_argument("--no_plot", action="store_true", help="Disable saving training curves")

    args = parser.parse_args()
    set_seed(args.seed)
    nltk.download("punkt", quiet=True)

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        captions_file=args.captions_file,
        images_root=args.images_root,
        image_transform=image_transform,
        batch_size=args.batch_size,
        min_word_freq=args.min_word_freq,
        split_seed=args.seed,
        num_workers=args.num_workers,
    )

    model = CaptioningModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_dim=args.encoder_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_bleu1": []}
    best_val_loss = float("inf")
    best_val_bleu1 = 0.0
    best_ckpt_path = os.path.join(args.results_dir, f"{args.experiment_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pad_idx=vocab.pad_idx,
        )
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=vocab.pad_idx,
        )
        val_bleu1 = evaluate_bleu1(model=model, dataloader=val_loader, vocab=vocab, device=device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu1"].append(val_bleu1)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bleu1={val_bleu1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu1 = max(best_val_bleu1, val_bleu1)
            checkpoint = {
                "model_state": model.state_dict(),
                "vocab_itos": vocab.itos,
                "config": {
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "encoder_dim": args.encoder_dim,
                    "freeze_encoder": args.freeze_encoder,
                },
            }
            torch.save(checkpoint, best_ckpt_path)

    # Optional test set summary with final model state
    test_loss = evaluate_loss(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        pad_idx=vocab.pad_idx,
    )
    test_bleu1 = evaluate_bleu1(model=model, dataloader=test_loader, vocab=vocab, device=device)

    metrics = {
        "experiment_name": args.experiment_name,
        "dataset_name": "flickr8k",
        "device": str(device),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "encoder_dim": args.encoder_dim,
        "min_word_freq": args.min_word_freq,
        "seed": args.seed,
        "vocab_size": len(vocab),
        "best_val_loss": float(best_val_loss),
        "best_val_bleu1": float(best_val_bleu1),
        "test_loss": float(test_loss),
        "test_bleu1": float(test_bleu1),
        "best_checkpoint": best_ckpt_path,
    }

    metrics_path = os.path.join(args.results_dir, f"{args.experiment_name}_metrics.json")
    history_csv_path = os.path.join(args.results_dir, f"{args.experiment_name}_history.csv")
    plot_path = os.path.join(args.results_dir, f"{args.experiment_name}_curves.png")
    log_path = os.path.join(args.results_dir, "experiments_log.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_history_csv(history, history_csv_path)

    if not args.no_plot:
        plot_history(history, plot_path)

    append_experiment_log(
        path=log_path,
        experiment_name=args.experiment_name,
        params={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "min_word_freq": args.min_word_freq,
            "seed": args.seed,
        },
        best_val_loss=best_val_loss,
        best_val_bleu1=best_val_bleu1,
    )

    print("\nFinal Summary")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Best Val BLEU1: {best_val_bleu1:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test BLEU1    : {test_bleu1:.4f}")
    print(f"Best model saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
