from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Reacher-v2',
'n_environment_steps': 1500000,
'group': 'poly_sigmoid_c4',
}
base.update(run_config)
