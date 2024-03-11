import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
from project.features.bitcoin_price import parse_btc, time_series_transforms_btc
from project.utils import (
    MLFLOW_TRACKING_URI,
    build_model,
    convert_date_to_unix,
    fetch_data_from_minio,
    fix_unix,
    prepare_train_test_dataset,
    scale_dataset,
    split_dataset,
    upload_minio,
)

load_dotenv()
with DAG(
    dag_id="ML_Data_DAG",
    default_args={"owner": "Baraa", "start_date": days_ago(1)},
    schedule_interval=None,
    catchup=False,
) as dag:
    #### LOAD DATA ####
    ###################
    def load_btc(**kwargs):
        btc_df = fetch_data_from_minio("datasets/1INCH-BTC.csv")
        return btc_df

    def load_tweets_textblob(**kwargs):
        tweets_textblob_df = fetch_data_from_minio("datasets/tweets_textblob.csv")
        tweets_textblob_df["date"] = tweets_textblob_df["date"].apply(lambda x: x[:10])
        return tweets_textblob_df

    def load_tweets_vader(**kwargs):
        tweets_vader_df = fetch_data_from_minio("datasets/tweets_vader.csv")
        tweets_vader_df["date"] = tweets_vader_df["date"].apply(lambda x: x[:10])
        return tweets_vader_df

    #### PREPROCESS DATA ####
    ########################

    #### PREPROCESS BTC DATA ####

    def convert_to_days(**kwargs):
        processed_btc_df = kwargs["ti"].xcom_pull(task_ids="load_btc")
        processed_btc_df = processed_btc_df.rename(columns={"open_time": "date"})
        processed_btc_df["date"] = pd.to_datetime(processed_btc_df["date"])
        processed_btc_df = processed_btc_df.set_index("date").resample("D").asfreq()
        processed_btc_df = processed_btc_df.reset_index().dropna()
        # processed_btc_df = processed_btc_df.tail(57).reset_index(drop=True)
        return processed_btc_df

    def parse_btc_data(**kwargs):
        btc_df = kwargs["ti"].xcom_pull(task_ids="convert_to_days")
        parsed_btc_df = parse_btc(btc_df)
        parsed_btc_df.reset_index(drop=True, inplace=True)

        return parsed_btc_df

    def preprocess_btc_data(**kwargs):
        parsed_btc_df = kwargs["ti"].xcom_pull(task_ids="parse_btc_data")
        processed_btc_df = time_series_transforms_btc(parsed_btc_df)
        return processed_btc_df

    def fix_btc_date(**kwargs):
        processed_btc_df = kwargs["ti"].xcom_pull(task_ids="preprocess_btc_data")
        processed_btc_df["unix"] = processed_btc_df["date"].apply(convert_date_to_unix)
        processed_btc_df["unix"] = processed_btc_df.unix.apply(fix_unix)
        processed_btc_df["date"] = processed_btc_df["date"].astype(str)
        return processed_btc_df

    def get_btc_dates(**kwargs):  # Used to fill the missing dates in the tweets data
        processed_btc_df = kwargs["ti"].xcom_pull(task_ids="fix_btc_date")
        dates = (
            processed_btc_df["date"]
            .sort_values()
            .reset_index(drop=True)
            .astype(str)
            .tolist()
        )
        return dates

    #### PREPROCESS TWEETS TEXTBLOB DATA ####

    def get_tweets_dates(**kwargs):
        tweets_textblob_df = kwargs["ti"].xcom_pull(task_ids="load_tweets_textblob")
        tweets_dates = (
            tweets_textblob_df["date"]
            .sort_values()
            .reset_index(drop=True)
            .astype(str)
            .tolist()
        )
        return tweets_dates

    def find_missing_dates(**kwargs):
        btc_dates = kwargs["ti"].xcom_pull(task_ids="get_btc_dates")
        tweets_dates = kwargs["ti"].xcom_pull(task_ids="get_tweets_dates")
        missing_dates = list(set(btc_dates) - set(tweets_dates))
        print("Num of missing dates:", len(missing_dates))
        return missing_dates

    def missing_dates_tweets_textblob(**kwargs):
        missing_dates = kwargs["ti"].xcom_pull(task_ids="find_missing_dates")
        tweets_textblob_missing_dates = pd.DataFrame(
            {
                "date": missing_dates,
                "subjectivity": [1] * len(missing_dates),
                "polarity": [1] * len(missing_dates),
            }
        )
        tweets_textblob_missing_dates["unix"] = tweets_textblob_missing_dates[
            "date"
        ].apply(convert_date_to_unix)
        return tweets_textblob_missing_dates

    def fill_missing_dates_tweets_textblob(**kwargs):
        tweets_textblob_df = kwargs["ti"].xcom_pull(task_ids="load_tweets_textblob")
        tweets_textblob_missing_dates = kwargs["ti"].xcom_pull(
            task_ids="missing_dates_tweets_textblob"
        )
        filled_tweets_textblob_df = (
            pd.concat([tweets_textblob_df, tweets_textblob_missing_dates])
            .sort_values("date")
            .reset_index(drop=True)
        )
        return filled_tweets_textblob_df

    #### PREPROCESS TWEETS VADER DATA ####
    def missing_dates_tweets_vader(**kwargs):
        missing_dates = kwargs["ti"].xcom_pull(task_ids="find_missing_dates")
        tweets_vader_missing_dates = pd.DataFrame(
            {
                "date": missing_dates,
                "compound": [1] * len(missing_dates),
            }
        )
        tweets_vader_missing_dates["unix"] = tweets_vader_missing_dates["date"].apply(
            convert_date_to_unix
        )
        return tweets_vader_missing_dates

    def fill_missing_dates_tweets_vader(**kwargs):
        tweets_vader_df = kwargs["ti"].xcom_pull(task_ids="load_tweets_vader")
        tweets_vader_missing_dates = kwargs["ti"].xcom_pull(
            task_ids="missing_dates_tweets_vader"
        )
        filled_tweets_vader_df = (
            pd.concat([tweets_vader_df, tweets_vader_missing_dates])
            .sort_values("date")
            .reset_index(drop=True)
        )
        return filled_tweets_vader_df

    def join_and_process_data(**kwargs):
        btc_df = kwargs["ti"].xcom_pull(task_ids="fix_btc_date")
        tweets_textblob = kwargs["ti"].xcom_pull(
            task_ids="fill_missing_dates_tweets_textblob"
        )
        tweets_textblob = tweets_textblob[["subjectivity", "polarity"]]

        tweets_vader = kwargs["ti"].xcom_pull(
            task_ids="fill_missing_dates_tweets_vader"
        )
        tweets_vader = tweets_vader[["compound"]]

        processed_final_data = pd.concat(
            [btc_df, tweets_textblob, tweets_vader], axis=1
        )

        return processed_final_data

    def scale_data(**kwargs):
        processed_final_data = kwargs["ti"].xcom_pull(task_ids="join_and_process_data")
        scaled_data = scale_dataset(processed_final_data)
        return scaled_data

    def upload_data_to_minio(**kwargs):
        scaled_data = kwargs["ti"].xcom_pull(task_ids="scale_data")
        upload_minio(scaled_data, destination_file="datasets/clean/final_df.csv")

    def train_model(**kwargs):
        import tensorflow as tf
        from mlflow.tensorflow import MLflowCallback

        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # mlflow.tensorflow.autolog()
        scaled_data = kwargs["ti"].xcom_pull(task_ids="scale_data")
        X_train, X_test, y_train, y_test = split_dataset(scaled_data)
        training_dataset, test_dataset = prepare_train_test_dataset(
            X_train, y_train, X_test, y_test
        )

        with mlflow.start_run() as run:
            model = model = build_model([1, X_train.shape[1]])
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",  # Monitor validation loss
                patience=10,  # Stop training after 10 epochs with no improvement
                min_delta=0.001,  # Minimum change in monitored quantity to be considered improvement
                restore_best_weights=True,  # Restore model weights from the best epoch
            )
            history = model.fit(
                training_dataset,
                epochs=10,
                verbose=0,
                steps_per_epoch=500,
                validation_data=test_dataset,
                validation_steps=1,
                callbacks=[MLflowCallback(run), early_stopping],
            )

            history_dict = history.history
            loss_values = history_dict["mae"]
            val_loss_values = history_dict["val_mae"]
            print(loss_values, val_loss_values)
            mlflow.tensorflow.log_model(
                model, "saved_model", keras_model_kwargs={"save_format": "h5"}
            )

    #### TASKS ####
    ###############

    #### BTC TASKS ####
    load_btc_task = PythonOperator(
        task_id="load_btc",
        python_callable=load_btc,
        provide_context=True,
    )

    convert_to_days_task = PythonOperator(
        task_id="convert_to_days",
        python_callable=convert_to_days,
        provide_context=True,
    )

    parse_btc_data_task = PythonOperator(
        task_id="parse_btc_data",
        python_callable=parse_btc_data,
        provide_context=True,
    )

    preprocess_btc_data_task = PythonOperator(
        task_id="preprocess_btc_data",
        python_callable=preprocess_btc_data,
        provide_context=True,
    )

    fix_btc_date_task = PythonOperator(
        task_id="fix_btc_date",
        python_callable=fix_btc_date,
        provide_context=True,
    )

    get_btc_dates_task = PythonOperator(
        task_id="get_btc_dates",
        python_callable=get_btc_dates,
        provide_context=True,
    )

    #### TWEETS TEXTBLOB TASKS ####
    load_tweets_textblob_task = PythonOperator(
        task_id="load_tweets_textblob",
        python_callable=load_tweets_textblob,
        provide_context=True,
    )
    get_tweets_dates_task = PythonOperator(
        task_id="get_tweets_dates",
        python_callable=get_tweets_dates,
        provide_context=True,
    )

    find_missing_dates_task = PythonOperator(
        task_id="find_missing_dates",
        python_callable=find_missing_dates,
        provide_context=True,
    )

    missing_dates_tweets_textblob_task = PythonOperator(
        task_id="missing_dates_tweets_textblob",
        python_callable=missing_dates_tweets_textblob,
        provide_context=True,
    )

    fill_missing_dates_tweets_textblob_task = PythonOperator(
        task_id="fill_missing_dates_tweets_textblob",
        python_callable=fill_missing_dates_tweets_textblob,
        provide_context=True,
    )

    #### TWEETS VADER TASKS ####
    load_tweets_vader_task = PythonOperator(
        task_id="load_tweets_vader",
        python_callable=load_tweets_vader,
        provide_context=True,
    )

    missing_dates_tweets_vader_task = PythonOperator(
        task_id="missing_dates_tweets_vader",
        python_callable=missing_dates_tweets_vader,
        provide_context=True,
    )

    fill_missing_dates_tweets_vader_task = PythonOperator(
        task_id="fill_missing_dates_tweets_vader",
        python_callable=fill_missing_dates_tweets_vader,
        provide_context=True,
    )

    join_and_process_data_task = PythonOperator(
        task_id="join_and_process_data",
        python_callable=join_and_process_data,
        provide_context=True,
    )

    scale_data_task = PythonOperator(
        task_id="scale_data",
        python_callable=scale_data,
        provide_context=True,
    )

    upload_data_to_minio_task = PythonOperator(
        task_id="upload_data_to_minio",
        python_callable=upload_data_to_minio,
        provide_context=True,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )
    # Set task dependencies
    (
        load_btc_task
        >> convert_to_days_task
        >> parse_btc_data_task
        >> preprocess_btc_data_task
        >> fix_btc_date_task
        >> join_and_process_data_task
    )

    fix_btc_date_task >> get_btc_dates_task
    (
        load_tweets_textblob_task
        >> [
            get_tweets_dates_task,
            get_btc_dates_task
            >> find_missing_dates_task
            >> missing_dates_tweets_textblob_task,
        ]
        >> fill_missing_dates_tweets_textblob_task
        >> join_and_process_data_task
    )

    (
        [load_tweets_vader_task >> find_missing_dates_task]
        >> missing_dates_tweets_vader_task
        >> fill_missing_dates_tweets_vader_task
        >> join_and_process_data_task
    )
    (
        join_and_process_data_task
        >> scale_data_task
        >> upload_data_to_minio_task
        >> train_model_task
    )
