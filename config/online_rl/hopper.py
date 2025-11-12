from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Hopper-v3',
'noise_sched_tau': 0.1,
'n_environment_steps': 1500000,
'guidance_type': 'grad',
'group': 'poly_exp_c4',
}
base.update(run_config)
