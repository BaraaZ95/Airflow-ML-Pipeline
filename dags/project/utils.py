import datetime
import io
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

load_dotenv()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def scale_dataset(df):
    unix_column = df["unix"]
    data_to_scale = df.drop(columns=["unix", "date"])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # Combine 'unix' column with scaled data
    scaled_data_with_unix = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
    scaled_data_with_unix["unix"] = unix_column.reset_index(drop=True)

    return scaled_data_with_unix


def split_dataset(df):
    target_column = "close"

    tscv = TimeSeriesSplit(n_splits=5)

    train_end = int(np.percentile(df["unix"], 80))
    test_start = int(np.percentile(df["unix"], 81))

    for train_index, test_index in tscv.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        train_data = train_data[train_data["unix"] <= train_end]
        test_data = test_data[test_data["unix"] >= test_start]

    X_train = train_data.drop(target_column, axis=1)
    X_test = test_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    y_test = test_data[target_column]
    X_train.drop(columns=["unix"], inplace=True)
    X_test.drop(columns=["unix"], inplace=True)
    return X_train, X_test, y_train, y_test


def windowed_dataset(dataset, target, window_size, batch_size):
    ds = dataset.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(window_size))
    ds = ds.map(lambda window: tf.reshape(window[-1:], [-1, 32]))

    target_ds = target.window(window_size, shift=1, drop_remainder=True)
    target_ds = target_ds.flat_map(lambda window: window.batch(window_size))
    target_ds = target_ds.map(lambda window: window[-1:])

    ds = tf.data.Dataset.zip((ds, target_ds))
    ds = ds.batch(batch_size, True)
    ds = ds.prefetch(1)
    return ds


def prepare_train_test_dataset(X_train, y_train, X_test, y_test):
    training_dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_train.values, tf.float32)
    )
    training_target = tf.data.Dataset.from_tensor_slices(
        np.array(y_train).flatten().tolist()
    )
    training_dataset = training_dataset.repeat(500)
    training_dataset = windowed_dataset(
        training_dataset, training_target, window_size=2, batch_size=16
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_test.values, tf.float32)
    )
    validation_target = tf.data.Dataset.from_tensor_slices(
        np.array(y_test).flatten().tolist()
    )
    training_dataset = training_dataset.repeat(500)
    test_dataset = windowed_dataset(
        test_dataset, validation_target, window_size=2, batch_size=16
    )
    return training_dataset, test_dataset


def build_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim[0], input_dim[1]))
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=1, padding="same", kernel_initializer="uniform"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = tf.keras.layers.Conv1D(
        filters=input_dim[1],
        kernel_size=1,
        padding="same",
        kernel_initializer="uniform",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="uniform")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation="relu", kernel_initializer="uniform")(x)

    model = tf.keras.Model(inputs, x)
    model.summary()

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


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


def upload_minio(df: pd.DataFrame, destination_file: str) -> None:
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_ACCESS_KEY,
            secure=False,
        )

        found = minio_client.bucket_exists(MINIO_BUCKET_NAME)
        if not found:
            minio_client.make_bucket(MINIO_BUCKET_NAME)
            print("Created bucket", MINIO_BUCKET_NAME)
        else:
            print("Bucket", MINIO_BUCKET_NAME, "already exists")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        csv_buffer = io.BytesIO(csv_bytes)

        minio_client.put_object(
            MINIO_BUCKET_NAME,
            destination_file,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type="application/csv",
        )
        print(f"Uploaded to {destination_file} at bucket: {MINIO_BUCKET_NAME}")
        return None

    except S3Error as exc:
        print("error occurred.", exc)
        return None


# def fetch_data_from_minio(dataset_path):
#    df = pd.read_csv(
#        f"s3://{MINIO_BUCKET_NAME}/{dataset_path}",
#        storage_options={
#            "key": MINIO_ROOT_USER,
#            "secret": MINIO_ROOT_PASSWORD,
#            "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
#        },
#    )
#    return df


def fetch_data_from_minio(object_name: str):
    minio_client = Minio(
        MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_ACCESS_KEY, secure=False
    )

    try:
        # Get the CSV object from MinIO
        csv_object = minio_client.get_object("s3-bucket", object_name)

        # Read the CSV data into a pandas DataFrame
        csv_data = io.BytesIO(csv_object.read())
        df = pd.read_csv(csv_data)

        return df
    except S3Error as err:
        print(err)
        return None
