import subprocess
import sys


EXPERIMENTS = [
    {
        "name": "cap_exp1_baseline",
        "learning_rate": 0.001,
        "epochs": 12,
        "batch_size": 32,
        "embed_dim": 256,
        "hidden_dim": 512,
        "min_word_freq": 5,
        "seed": 42,
    },
    {
        "name": "cap_exp2_higher_hidden",
        "learning_rate": 0.001,
        "epochs": 12,
        "batch_size": 32,
        "embed_dim": 256,
        "hidden_dim": 768,
        "min_word_freq": 5,
        "seed": 42,
    },
    {
        "name": "cap_exp3_lower_lr",
        "learning_rate": 0.0005,
        "epochs": 15,
        "batch_size": 32,
        "embed_dim": 256,
        "hidden_dim": 512,
        "min_word_freq": 3,
        "seed": 42,
    },
]


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python src/run_experiments.py <captions_file> <images_root>\n"
            "Example: python src/run_experiments.py data/Flickr8k.token.txt data/Images"
        )
        sys.exit(1)

    captions_file = sys.argv[1]
    images_root = sys.argv[2]

    for exp in EXPERIMENTS:
        cmd = [
            sys.executable,
            "src/train.py",
            "--captions_file",
            captions_file,
            "--images_root",
            images_root,
            "--experiment_name",
            exp["name"],
            "--learning_rate",
            str(exp["learning_rate"]),
            "--epochs",
            str(exp["epochs"]),
            "--batch_size",
            str(exp["batch_size"]),
            "--embed_dim",
            str(exp["embed_dim"]),
            "--hidden_dim",
            str(exp["hidden_dim"]),
            "--min_word_freq",
            str(exp["min_word_freq"]),
            "--seed",
            str(exp["seed"]),
            "--freeze_encoder",
        ]

        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)

        if completed.returncode != 0:
            print(f"Experiment {exp['name']} failed with code {completed.returncode}")
            break


if __name__ == "__main__":
    main()
