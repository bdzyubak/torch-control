from conda_utils import check_conda_installed, execute_setup_py_in_conda_env, develop_submodules, conda_create, \
    install_hard_linked_pytorch
from utils.os_utils import run_command


def main():
    # Use this entrypoint to install shared dependencies in the base conda environment and avoid duplication.
    # This can only be used for dependencies with pinned versions e.g. opendatasets. To conserve space, I am using a
    # shared torch version (i.e. I've switched requirements.txts to have >=2.0). For production stability you would
    # want to pin specific versions of all dependencies.

    check_conda_installed()

    # TODO: develop command line inputs for which envs to install and flags
    envs = {'nnunet': {'install_method': 'setup_py', 'submodule_name': 'nnUNet'}}
    develop_paths = False
    clean_install = False

    # Pip must be first in list to use pip install fallback. Otherwise, global linked pip is used and everything
    # deps install to base
    shared_deps = ['pip', 'opendatasets', 'hiddenlayer', 'seaborn', 'openai', 'transformers']

    for env_name in envs:
        if clean_install:
            run_command(f'conda remove -n {env_name}')

        env_name = conda_create(env_name=env_name)

        install_shared_dependencies(env_name, shared_deps)

        if envs[env_name]['install_method'] == 'setup_py':
            install_hard_linked_pytorch(env_name)
            execute_setup_py_in_conda_env(env_name, submodule_name=envs[env_name]['submodule_name'])
        else:
            # Add direct pip install and custom conda/pip/extra wheels combo, when needed.
            raise NotImplementedError(f"Method {envs[env_name]['install_method']} is not implemented, yet.")

        if develop_paths:
            develop_submodules(env_name)


def install_shared_dependencies(env_name, shared_deps):
    # Install dependencies that are not part of submodule requirements but are useful for running torch-control
    # experiments without switching to a dedicated interpreter.  Using pip to install since some deps are not on the
    # conda default channel
    print(f'Installing shared deps to {env_name}.')
    install_failed = dict()
    for shared_dep in shared_deps:
        # Use conda to hard link and conserve space
        command_install = f'conda install -n {env_name} {shared_dep} -y'
        retcode, text = run_command(command_install, verbose=verbose)

        if retcode != 0:
            # Conda install failed, fall back to pip inside conda env
            # NB: If you did not do conda install -n [env_name] pip, deps will install to base!
            command_install = f'conda run -n {env_name} python -m pip install {shared_dep}'
            retcode, text = run_command(command_install, verbose=verbose)
            if retcode != 0:
                install_failed[shared_dep] = text
    if install_failed:
        raise OSError(f"Failed to install dependencies {install_failed}")


if __name__ == "__main__":
    verbose = True
    main()
