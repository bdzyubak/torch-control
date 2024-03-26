import argparse
from pathlib import Path


from conda_utils import (check_conda_installed, develop_submodules,
                         conda_create_from_yml, conda_extend_env)
from utils.os_utils import run_command


def main(envs_list=None):
    # Use this entrypoint to install shared dependencies in the base conda environment and avoid duplication.
    # This can only be used for dependencies with pinned versions e.g. opendatasets. To conserve space, I am using a
    # shared torch version (i.e. I've switched requirements.txts to have >=2.0). For production stability you would
    # want to pin specific versions of all dependencies.

    check_conda_installed()
    path_dependencies = Path(__file__).parent

    envs = {'cv': {'install_method': 'environment_computer_vision.yml', 'submodules': ['nnUNet']},
            'nlp': {'install_method': 'environment_natural_language_processing.yml', 'submodules': []},
            'ml': {'install_method': 'environment_machine_learning.yml', 'submodules': []}}
    if envs_list is not None:
        envs = {key: value for key, value in envs.items() if key in envs_list}
    develop_paths = False

    # Install shared torch in base environment. Other environments will be able to see it if their required
    # version is the same
    run_command(f"pip install -r {path_dependencies / 'requirements.txt'}")

    for env_name in envs:
        run_command(f'conda remove -n {env_name} --all --y')

        conda_create_from_yml(env_name=env_name, file=path_dependencies / envs[env_name]['install_method'])

        if develop_paths:
            develop_submodules(env_name)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs_list', nargs='+', default=None,
                        help="Which command line args to install")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_command_line_args()
    main(envs_list=args.envs_list)
