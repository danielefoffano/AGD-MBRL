from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Pusher-v2',
'n_environment_steps': 1500000,
'norm_keys': ['actions', 'rewards', 'terminals'],
}
base.update(run_config)
