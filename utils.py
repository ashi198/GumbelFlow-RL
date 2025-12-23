import os, mlflow 


def set_mlflow_connection():
    #os.environ["AWS_ACCESS_KEY_ID"] = "minio_id"
    #os.environ["AWS_SECRET_ACCESS_KEY"] = "XaFC8sHiRHr5uQTdfVQQ"
    #os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:5000" 
    #os.environ['MLFLOW_TRACKING_USERNAME'] = "hilbert" #"fhaselbeck"
    #os.environ['MLFLOW_TRACKING_PASSWORD'] = "schneckenfunktion"
    #os.environ['GIT_PYTHON_REFRESH'] ="quiet"
    remote_server_uri = "http://0.0.0.0:5000"  # "http://10.154.6.32:5100"  # set to MLFlow server URI (host ip and PORT in .env)
    mlflow.set_tracking_uri(remote_server_uri)
