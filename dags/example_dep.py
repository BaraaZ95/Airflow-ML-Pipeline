"""
Example DAG demonstrating the usage of the classic Python operators to execute Python functions natively and
within a virtual environment.
"""
import logging
import sys

import pendulum
from airflow import DAG
from airflow.operators.python import (
    ExternalPythonOperator,
    PythonVirtualenvOperator,
    is_venv_installed,
)

log = logging.getLogger(__name__)
PATH_TO_PYTHON_BINARY = sys.executable

with DAG(
    dag_id="example_python_operator",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
):
    if not is_venv_installed():
        log.warning(
            "The virtalenv_python example task requires virtualenv, please install it."
        )
    else:

        def callable_virtualenv():
            """
            Example function that will be performed in a virtual environment.

            Importing at the module level ensures that it will not attempt to import the
            library before it is installed.
            """
            import hopsworks

            project = hopsworks.login(
                api_key_value="kmUzHAqgN1ST77pR.vNxzfcPuUfwHgrC8B5XhXAtUGRqasAJddHAJ2WricYkC1PZjGqPVMGZJugqF1DFg"
            )

            fs = project.get_feature_store()
            tweets_textblob_fg = fs.get_or_create_feature_group(
                name="bitcoin_tweets_textblob",
                version=1,
            )
            stored_tweets_df = tweets_textblob_fg.read()
            print(stored_tweets_df.head())

        virtualenv_task = PythonVirtualenvOperator(
            task_id="virtualenv_python",
            python_callable=callable_virtualenv,
            requirements=[
                "hopsworks",
            ],
            system_site_packages=False,
        )

        def callable_external_python():
            """
            Example function that will be performed in a virtual environment.

            Importing at the module level ensures that it will not attempt to import the
            library before it is installed.
            """

        external_python_task = ExternalPythonOperator(
            task_id="external_python",
            python_callable=callable_external_python,
            python=PATH_TO_PYTHON_BINARY,
        )

        external_python_task >> virtualenv_task
