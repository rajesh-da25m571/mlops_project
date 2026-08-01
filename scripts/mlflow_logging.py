import os
import sys
import json
import mlflow

# Configuration Constants
MANIFEST_PATH = "/opt/airflow/data/ocr_batch_manifest.json"
IMAGES_DIR = "/opt/airflow/data/output_images"
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"
EXPERIMENT_NAME = "mlops_training_flow"

MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")
#MLFLOW_TRACKING_URI = f"http://docker.internal:{MLFLOW_PORT}"
#EXPERIMENT_NAME = "mlops_training_flow"

def sync_to_mlflow():
    # Verify if a manifest exists
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest file not found at {MANIFEST_PATH}. Skipping log run.")
        sys.exit(0)

    # Load data logged by the WSL processing script
    with open(MANIFEST_PATH, 'r') as f:
        runs_to_log = json.load(f)

    try:
        # Establish connection with the MLflow Server
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e :
        print("Unable to connect to MLFlow server ",e)

    print(f"Syncing {len(runs_to_log)} entries into experiment '{EXPERIMENT_NAME}'...")

    for run_data in runs_to_log:
        run_name = f"Log_{run_data['file_name']}"
        
        # Start a distinct MLflow run tracking sequence
        with mlflow.start_run(run_name=run_name):
            # Log raw metadata variables
            mlflow.log_param("file_name", run_data['file_name'])
            mlflow.log_metric("img_width", run_data['img_width'])
            mlflow.log_metric("img_height", run_data['img_height'])
            mlflow.log_metric("avg_ocr_confidence", run_data['avg_ocr_confidence'])
            mlflow.log_metric("extracted_blocks_count", run_data['extracted_blocks_count'])
            
            # Extract and upload the visual bounding box image saved by OpenCV
            wsl_path = run_data['saved_artifact_path']
            #change wsl path to container path
            local_img_path = wsl_path.replace("/mnt/d/mlops_project/data", "/opt/airflow/data")
            if os.path.exists(local_img_path):
                #windows_path = wsl_path.replace("/mnt/d/", "D:\\").replace("/", "\\") # Convert to Windows format
                mlflow.set_tag("local_visualization_path", wsl_path)
                #mlflow.log_artifact(local_img_path, artifact_path="bounding_box_visualizations")
                print(f" Successfully logged run and artifacts for: {run_data['file_name']}")
            else:
                print(f" Warning: Image artifact missing at {local_img_path}")

    # clear contents of manifest file
    #with open(MANIFEST_PATH, 'w') as f:
    #    json.dump([], f)
    #os.remove(MANIFEST_PATH)
    print("Sync process completed. Manifest cleared.")

if __name__ == "__main__":
    sync_to_mlflow()
