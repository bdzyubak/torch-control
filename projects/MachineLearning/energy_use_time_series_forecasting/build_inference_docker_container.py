
from pathlib import Path
import mlflow

from dataloader import get_energy_use_data
from docker_utils import add_gcc_to_dockerfile
from services.dataframe_analysis.time_series import get_accuracy_metrics_df
from os_utils import run_command


# Get this from gridsearch logged to mlflow ui at localhost:8080
# Optimize xgboost: [parent_experiment] / [artifacts] / [best_estimator]

# Can do run
# logged_model = 'runs:/aba1c2cb5d0f4ab6876236520dcf2706/best_estimator'
# But better to do registered model name and version. "Staging" syntax is deprecated but alias can be used instead
logged_model = 'models:/PowerPrediction-xgboost-v1/2'

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = get_energy_use_data()
X_val = data['X_val']

# Validate model wrt logs
X_val['prediction'] = loaded_model.predict(X_val)
X_val, val_rmse = get_accuracy_metrics_df(X_val, data['y_val'])

# Manually enter the expected RMSE value from the release documentation (NOT the mlflow run above) as a double check
# that the right model is being released
assert val_rmse == 1819, 'RMSE does not match value specified in release documentation'

# Predict on a Pandas DataFrame.
X_val_pred = loaded_model.predict(data['X_val'])

model_dir = Path(r"D:\Models\ML\powerprediction_v0p2")
model_dir.mkdir(exist_ok=True, parents=True)

# Fails due to missing GCC
# run_command(f"mlflow models build-docker --model-uri {logged_model} --name {container_name}")

# Make docker file, add gcc, and build manually
run_command(f"mlflow models generate-dockerfile --model-uri {logged_model} --output-directory {model_dir}")
add_gcc_to_dockerfile(model_dir=model_dir)

image_name = model_dir.stem.split('_v')[0]
version = 'v'+model_dir.stem.split('_v')[1]
run_command(f"docker build -t {image_name}:{version} {model_dir}")


