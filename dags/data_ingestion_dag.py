from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator

# Environment configuration paths
IMAGE_FOLDER_DIR = "/mnt/d/mlops_project/input_images"
PROCESSOR_SCRIPT = "/mnt/d/mlops_project/scripts/ocr_preprocessing.py"
#LOGGER_SCRIPT = "/mnt/d/mlops_project/scripts/mlflow_logging.py"
LOGGER_SCRIPT = "/opt/airflow/scripts/mlflow_logging.py"

wsl_ssh_hook = SSHHook(
    remote_host="host.docker.internal",
    username="rajesh",
    password="admin123",
    port=22,
)

default_args = {
    'owner': 'mlops_platform',
    'start_date': datetime(2026, 1, 1),
    'catchup': False
}

with DAG(
    dag_id="decoupled_ocr_mlflow_pipeline",
    default_args=default_args,
    schedule=None,
    tags=["lightweight_core", "architecture_optimized"]
) as dag:

    # Task 1: Runs natively inside WSL Python environment
    run_ml_task = SSHOperator(
        task_id="run_ml_model",
        #ssh_conn_id="wsl_host_ssh",
        ssh_hook=wsl_ssh_hook,
        # 1. Source the virtual environment
        command=(
            "source /home/rajesh/ocr_env/bin/activate && "
            "python /mnt/d/mlops_project/scripts/ocr_preprocessing.py /mnt/d/mlops_project/input_images"
        ),
        cmd_timeout=3600, # Adjust based on how long your ML job takes
    )


    # Task 2: Runs inside the Airflow Docker Worker using the container's standard python engine
    # (Requires only 'mlflow' installed)
    run_docker_mlflow_logger = BashOperator(
        task_id="sync_artifacts_to_mlflow",
        bash_command=f"python3 {LOGGER_SCRIPT}"
    )

    # Sequential pipeline flow map
    run_ml_task >> run_docker_mlflow_logger
