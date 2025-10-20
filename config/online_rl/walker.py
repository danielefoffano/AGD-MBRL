from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Walker2d-v3',
'n_environment_steps': 1500000,
'guidance_type': 'grad',
'group': 'poly_sigmoid_c3',
}
base.update(run_config)
