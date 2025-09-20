import os
import gdown  # pip install gdown

DATA_DIR = "data"

def download_celeb_df():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Example Google Drive link (replace with real)
    url = "https://drive.google.com/uc?id=<FILE_ID>"
    output = os.path.join(DATA_DIR, "celeb_df_sample.zip")
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_celeb_df()
