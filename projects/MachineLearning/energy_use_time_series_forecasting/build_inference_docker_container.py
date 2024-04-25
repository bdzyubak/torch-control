from pathlib import Path
import mlflow
from subprocess import Popen
import time

from dataloader import get_energy_use_data
from docker_utils import convert_dataframe_to_json_for_docker, post_json_get_preds
from services.dataframe_analysis.time_series import get_accuracy_metrics_df
from os_utils import run_command, remove_dir

build_docker = True

# Script to pull down the registered model, build a docker container, and validate that inference results still match recorded baseline
#
# Eventually, update and use as part of CI/CD any time a model is registered or code is pushed
# Set a target accuracy below based on the validation docs NOT the model registry to check that the right model is being released

logged_model = 'models:/PowerPrediction-xgboost-v1/2'
target_rmse = 1633
client = mlflow.MlflowClient()

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = get_energy_use_data()
X_val = data['X_val'].copy()

# Validate model wrt logs
X_val['prediction'] = loaded_model.predict(X_val)
X_val, val_rmse = get_accuracy_metrics_df(X_val, data['y_val'])

# Manually enter the expected RMSE value from the release documentation (NOT the mlflow run above) as a double check
# that the right model is being released
assert val_rmse == target_rmse, (
    'mlflow class prediction does not match target. Wrong model loaded or target specified '
    'from wrong release?')
del X_val, val_rmse

model_dir = Path(r"D:\Models\ML\powerprediction_v0p2")
model_dir.mkdir(exist_ok=True, parents=True)

if build_docker:
    # Download model and docker file. If first time, modify Dockerfile by adding gcc and exposing ports,
    # and put in source tree
    remove_dir(model_dir)
    run_command(f"mlflow models generate-dockerfile --model-uri {logged_model} --output-directory {model_dir}")

    image_name = model_dir.stem.split('_v')[0]
    dockerfile_path = Path(__file__).parents[3] / 'dockerfiles' / 'Dockerfile_ML'
    # Last argument is the build context i.e. where stuff to copy into the container is
    run_command(f"docker build -t {image_name} -f {dockerfile_path} {model_dir}")
    remove_dir(model_dir)

docker_port_host = '8000'
p = Popen(["docker", "run", "-p", f"{docker_port_host}:5001", "powerprediction:latest"])

print('Paused to allow docker container to boot.')
time.sleep(5)

# Test the container
X_val = data['X_val'].copy()
X_val_json = convert_dataframe_to_json_for_docker(X_val)

# Spin up the docker container
preds = post_json_get_preds(X_val_json, docker_port_host)

X_val['prediction'] = preds
X_val, val_rmse = get_accuracy_metrics_df(X_val, data['y_val'])
assert val_rmse == target_rmse, "Prediction via docker container inaccurate. Check dependencies, right model, right port"

p.terminate()

## If no response is received from the docker container, debug via mlflow serve. This has better messages than once it
# is wrapped in docker.
# TODO: use python functions in place of CLI to add this to the automated testing workflow
# mlflow models serve --model-uri models:/PowerPrediction-xgboost-v1/2 --no-conda --port 5000 - successful
# Then use the following to curl either the local docker-less serve to make sure input data is interpreted correctly
# and predictions are received.
# Finally, Docker run the container:
# docker run -p 5001:8080 "powerprediction:v0p2"
# And curl the same command to the docker port to debug connectivity.
# curl -d "{\"inputs\":{\"dayofyear\":[1,2,3], \"hour\":[1,2,3], \"dayofweek\":[1,2,3], \"quarter\":[1,2,3], \"month\":[1,2,3], \"year\":[1,2,3]}}" -H "Content-Type: application/json"  http://127.0.0.1:8000/invocations
# Expected response - {"predictions": [12695.537109375, 12193.8173828125, 12634.658203125]}
# The following was necessary to get connectivity to work for me:
# ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent -b 0.0.0.0:5001" <- added -b 0.0.0.0:5001
# EXPOSE 5001
# ENV SERVER_PORT 8000
# ENV SERVER_HOST 0.0.0.0
