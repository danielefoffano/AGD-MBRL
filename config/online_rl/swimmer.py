from ..base_config import base, args_to_watch, logbase

run_config = {
'env_name':'Swimmer-v3',
'n_environment_steps': 1500000,
'noise_sched_tau': 0.1,
'lr_actor': 7e-4,
'lr_critic': 7e-4,
}
base.update(run_config)
