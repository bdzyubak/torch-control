
from pathlib import Path
import mlflow

from dataloader import get_energy_use_data
from docker_utils import add_gcc_to_dockerfile, convert_dataframe_to_json_for_docker, post_json_get_preds
from services.dataframe_analysis.time_series import get_accuracy_metrics_df
from os_utils import run_command

build_docker = False

# A refined model of this script should eventually be used for CI/CD - once a new model is registered as a release
# candidate or code is pushed to the release branch, run the following steps to test the model fetched from mlflow as a
# class, then test via mlflow serve, then test via Docker and make sure outputs on val data keep matching the reported
# accuracy in the validation documents

# Get this from gridsearch logged to mlflow ui at localhost:8080
# Optimize xgboost: [parent_experiment] / [artifacts] / [best_estimator]

# Can do run
# logged_model = 'runs:/aba1c2cb5d0f4ab6876236520dcf2706/best_estimator'
# But better to do registered model name and version. "Staging" syntax is deprecated but alias can be used instead
logged_model = 'models:/PowerPrediction-xgboost-v1/2'
target_rmse = 1633

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = get_energy_use_data()
X_val = data['X_val'].copy()

# Validate model wrt logs
X_val['prediction'] = loaded_model.predict(X_val)
X_val, val_rmse = get_accuracy_metrics_df(X_val, data['y_val'])

# Manually enter the expected RMSE value from the release documentation (NOT the mlflow run above) as a double check
# that the right model is being released
assert val_rmse == target_rmse, ('mlflow class prediction does not match target. Wrong model loaded or target specified '
                                 'from wrong release?')
del X_val, val_rmse

model_dir = Path(r"D:\Models\ML\powerprediction_v0p2")
model_dir.mkdir(exist_ok=True, parents=True)

## If steps below fail, debug via mlflow serve. This has better messages than once it is wrapped in docker.
# TODO: use python interfact to add this to the automated testing workflow
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

if build_docker:
    # Make docker file, add gcc, and build manually
    run_command(f"mlflow models generate-dockerfile --model-uri {logged_model} --output-directory {model_dir}")
    add_gcc_to_dockerfile(model_dir=model_dir)
    # Add step to insert EXPOSE 5001
    image_name = model_dir.stem.split('_v')[0]
    # version = 'v'+model_dir.stem.split('_v')[1]
    run_command(f"docker build -t {image_name} {model_dir}")
    # The build will drop the selected model as /opt/ml/model/MLmodel

# Test the container
X_val = data['X_val'].copy()
X_val_json = convert_dataframe_to_json_for_docker(X_val)

preds = post_json_get_preds(X_val_json)

X_val['prediction'] = preds
X_val, val_rmse = get_accuracy_metrics_df(X_val, data['y_val'])
assert val_rmse == target_rmse, "Prediction via docker container inaccurate. Check dependencies, right model, right port"
