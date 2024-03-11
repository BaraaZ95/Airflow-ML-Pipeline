# Airflow-ML-Pipeline
This is an example of setting up  MLOPs training Apache Airflow ML pipeline infrastructure  with docker-compose, MLFlow and minio. 

Note: This pipeline project is inspired by this [hopsworks tutorial](https://github.com/logicalclocks/hopsworks-tutorials/tree/master/advanced_tutorials/bitcoin). 


## Getting started 

1. First start the docker container by:

```bash
docker compose --env-file config.env up --build
```


2. Then  head over to the minio dashboard at http://localhost:9001 and generate access and secret keys. 
After that replace the secret keys in the `.env` located at `dags/project` and the `config.env` file at the root.


3. Create a .env file at dags/project with the following variables: 
```
MINIO_ENDPOINT = ...:9000
MINIO_ACCESS_KEY= 
MINIO_SECRET_ACCESS_KEY= 
MINIO_BUCKET_NAME= s3-bucket
MINIO_ROOT_USER: 'minio_user'
MINIO_ROOT_PASSWORD: 'minio_pwd'
MLFLOW_TRACKING_URI = ...:5001
```
If you are using wsl, the MINIO_ENDPOINT and MLFLOW_TRACKING_URI are your host ip-address which can be known by this bash command:

`ifconfig eth0 | grep 'inet ' | awk '{print $2}'`

For example it will be: `132.57.66.131`
Just copy past it at the beginning of the url: 
```
MINIO_ENDPOINT = 132.57.66.131:9000
MLFLOW_TRACKING_URI = 132.57.66.131:5001
```

4. Restart the docker-compose:
```
docker-compose down
docker compose --env-file config.env up --build
```


5. Upload the data to minio. Simply run `upload_data_minio.py`


