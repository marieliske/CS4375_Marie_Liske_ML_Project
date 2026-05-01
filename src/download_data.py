from pathlib import Path
import urllib.request
from zipfile import ZipFile

# downloads the flickr8k dataset from the github release and extracts it
# the zip is about 1GB so it takes a few minutes

DATASET_URL = "https://github.com/marieliske/Flickr8k-Dataset/releases/download/v1.0.0/flikr8k.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "flickr8k.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {DATASET_URL}")
    print("Please give a few minutes...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    print(f"Saved archive to: {ZIP_PATH}")
    print(f"Extracting to: {DATA_DIR}")

    with ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
