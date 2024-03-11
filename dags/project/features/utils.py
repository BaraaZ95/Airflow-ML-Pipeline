import datetime
import io
import os

import pandas as pd
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME")
MINIO_ROOT_USER = os.environ.get("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.environ.get("MINIO_ROOT_PASSWORD")


def get_hours(unix):
    return unix / 3600000 % 24


def fix_unix(unix):
    if get_hours(unix) == 23.0:
        return unix - 3600000
    elif get_hours(unix) == 21.0:
        return unix + 3600000
    return unix


def convert_unix_to_date(x):
    x //= 1000
    x = datetime.datetime.fromtimestamp(x)
    return datetime.datetime.strftime(x, "%Y-%m-%d")


def convert_date_to_unix(x):
    try:
        dt_obj = datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt_obj = datetime.datetime.strptime(str(x), "%Y-%m-%d")
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)


def upload_minio(source_file: str, destination_file: str, bucket_name: str) -> None:
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )

        found = minio_client.bucket_exists(bucket_name)
        if not found:
            minio_client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        else:
            print("Bucket", bucket_name, "already exists")

        minio_client.fput_object(
            bucket_name,
            destination_file,
            source_file,
        )
        print(
            source_file,
            "successfully uploaded as object",
            destination_file,
            "to bucket",
            bucket_name,
        )
        return None

    except S3Error as exc:
        print("error occurred.", exc)
        return None


# def fetch_data_from_minio(dataset_path):
#    df = pd.read_csv(
#        f"s3://test/{dataset_path}",
#        storage_options={
#            "key": "minio_user",
#            "secret": "minio_pwd",
#            "client_kwargs": {"endpoint_url": "http://172.17.42.151:9000"},
#        },
#    )
#    return df
#
#
# def fetch_data_from_minio(dataset_path):
#    df = pd.read_csv(
#        f"s3://test/{dataset_path}",
#        storage_options={
#            "key": "minio_user",
#            "secret": "minio_pwd",
#            "client_kwargs": {"endpoint_url": "http://172.17.42.151:9000"},
#        },
#    )
#    return df
#


def fetch_data_from_minio(object_name):
    # Initialize the MinIO client
    minio_client = Minio(
        MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=False
    )

    try:
        # Get the CSV object from MinIO
        csv_object = minio_client.get_object(MINIO_BUCKET_NAME, object_name)

        # Read the CSV data into a pandas DataFrame
        csv_data = io.BytesIO(csv_object.read())
        df = pd.read_csv(csv_data)

        return df
    except S3Error as err:
        print(err)
        return None
