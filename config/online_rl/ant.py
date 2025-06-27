from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Ant-v3',
'n_environment_steps': 1500000,
#'learning_rate': 1e-4,
'noise_sched_tau': 0.1,
}
base.update(run_config)
