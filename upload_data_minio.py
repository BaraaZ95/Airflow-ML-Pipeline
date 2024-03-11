from io import BytesIO

import pandas as pd
from minio import Minio, S3Error


def save_data(filepath, file_out):
    df = pd.read_csv(filepath)
    df.to_csv(file_out, index=False)


def convert_parquet_to_csv(filepath):
    df = pd.read_parquet(filepath)
    df.to_csv(filepath.replace("parquet", "csv"), index=False)


def upload_to_minio(filepath, object_name):
    minio_client = Minio(
        "127.0.0.1:9000", access_key="minio_user", secret_key="minio_pwd", secure=False
    )

    # Define bucket and object names
    bucket_name = "s3-bucket"

    df = pd.read_csv(filepath)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)

    try:
        minio_client.put_object(
            bucket_name,
            object_name,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type="application/csv",
        )
        print(f"uploaded {filepath} to {object_name}")
    except S3Error as err:
        print(err)


# Bitcoin dataset downloaded from https://www.kaggle.com/datasets/jorijnsmit/binance-full-history?select=1INCH-BTC.parquet
convert_parquet_to_csv("data/raw/1INCH-BTC.parquet")
upload_to_minio("data/raw/1INCH-BTC.csv", "datasets/1INCH-BTC.csv")

save_data(
    "https://repo.hops.works/dev/davit/bitcoin/tweets_textblob.csv",
    "data/raw/tweets_textblob.csv",
)
save_data(
    "https://repo.hops.works/dev/davit/bitcoin/tweets_vader.csv",
    "data/raw/tweets_vader.csv",
)
upload_to_minio("data/raw/tweets_textblob.csv", "datasets/tweets_textblob.csv")
upload_to_minio(
    "data/raw/tweets_vader.csv",
    "datasets/tweets_vader.csv",
)
