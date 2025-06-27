from polygrad.utils.evaluation import evaluate_policy
from polygrad.utils.envs import create_env
import torch
import numpy as np
import polygrad.utils as utils
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    config: str = "config.simple_maze"
    seed: int = 1
    run_number: int = 0


args = Parser().parse_args()

expl_env = create_env(args.env_name, args.suite)
eval_env = create_env(args.env_name, args.suite)
random_episodes = utils.rl.random_exploration(args.n_prefill_steps, expl_env)

run_nr = args.run_number
print(f"Run number {run_nr}")
print("Seed", args.seed)
utils.set_all_seeds(args.seed)
# load all config params

configs = utils.create_configs(args, eval_env)
if configs["render_config"] is not None:
    renderer = configs["render_config"]()
else:
    renderer = None
model = configs["model_config"]()
diffusion = configs["diffusion_config"](model)
value_model = configs["value_model_config"]()
value_diffusion = configs["value_diffusion_config"](value_model)
dataset = configs["dataset_config"](random_episodes)
diffusion_trainer = configs["trainer_config"](diffusion, dataset, eval_env, value_diffusion, renderer)
ac = configs["ac_config"](normalizer=dataset.normalizer)
agent = configs["agent_config"](
    diffusion_model=diffusion_trainer.ema_model,
    actor_critic=ac,
    dataset=dataset,
    env=eval_env,
    renderer=renderer,
    value_model = diffusion_trainer.ema_model_value
)

#joints = {"Hopper-v3": 3,
#          "HalfCheetah-v3": 6,
#          "Walker2d-v3": 7
#          }

new_paths = {"Hopper-v3": "Adversarial_Diff_Hopper",
          "HalfCheetah-v3": "Adversarial_Diff_Cheetah",
          "Walker2d-v3": "Adversarial_Diff_Walker",
          "InvertedPendulum-v4":"Adversarial_Diff_InvertedPendulum",
          "Reacher-v2":"Adversarial_Diff_Reacher"}

#joint = joints[args.env_name]
seed = args.seed
for seed in [1,2,4,5]:
    print(f"My seed {seed}")
    path = f"./scripts/logs/{new_paths[args.env_name]}/seed{seed}"
    step = 1500000#args.n_environment_steps
    agent.load(path, step, run=seed)

    eval_env = create_env(args.env_name, args.suite)
    eval_metrics = evaluate_policy(
                ac.forward_actor,
                eval_env,
                device,
                step,
                dataset,
                use_mean=True,
                n_episodes=10,
                renderer=renderer,
            )
    print(f"Metrics for training environment")
    print(eval_metrics)
#dampings = np.linspace(eval_env.sim.model.dof_damping[joint]/2, eval_env.sim.model.dof_damping[joint]*2, 50)

if args.env_name == "InvertedPendulum-v4":
    masses_cart = np.linspace(25, 1, 50)
    masses_pole = np.linspace(25, 1, 50)
    gravities = np.linspace(eval_env.env._env.env.env.env.model.opt.gravity[2]/2.0, eval_env.env._env.env.env.env.model.opt.gravity[2]*2.0, 50)

    # csv_file_path = path+ f"/all_mass_cart_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Mass", "Avg_Return", "Std_Return"])
    #     for mass in masses_cart:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.env._env.env.env.env.model.body_mass[1] = mass
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([mass, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for cart mass {eval_env.env._env.env.env.env.model.body_mass[1]}")
    #         print(eval_metrics)

    # csv_file_path = path+ f"/all_mass_pole_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Mass", "Avg_Return", "Std_Return"])
    #     for mass in masses_pole:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.env._env.env.env.env.model.body_mass[2] = mass
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([mass, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for pole mass {eval_env.env._env.env.env.env.model.body_mass[2]}")
    #         print(eval_metrics)

    csv_file_path = path+ f"/all_gravity_Adversarial_Diff_avg_returns-{seed}.csv"

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Gravity", "Avg_Return", "Std_Return"])
        for gravity in gravities:
            eval_env = create_env(args.env_name, args.suite)
            eval_env.env._env.env.env.env.model.opt.gravity[2] = gravity
            eval_metrics = evaluate_policy(
                        ac.forward_actor,
                        eval_env,
                        device,
                        step,
                        dataset,
                        use_mean=True,
                        n_episodes=10,
                        renderer=renderer,
                    )
            csv_writer.writerow([gravity, eval_metrics["avg_return"], eval_metrics["std_return"]])
            print(f"Metrics for gravity {eval_env.env._env.env.env.env.model.opt.gravity[2]}")
            print(eval_metrics)
elif args.env_name == "Reacher-v2":

    ac_gears = np.linspace(50, 1000, 50)
    frictionlosses = np.linspace(0, 150, 50)

    # csv_file_path = path+ f"/all_ac_gears_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["ac_gear", "Avg_Return", "Std_Return"])
    #     for ac_gear in ac_gears:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.sim.model.actuator_gear[:,0] = ac_gear
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([ac_gear, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for actuator gear {eval_env.sim.model.actuator_gear[:,0]}")
    #         print(eval_metrics)

    # csv_file_path = path+ f"/all_frictionlosses_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Frictionloss", "Avg_Return", "Std_Return"])
    #     for frictionloss in frictionlosses:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.sim.model.dof_frictionloss[[0,1]] = frictionloss
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([frictionloss, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for friction loss {eval_env.sim.model.dof_frictionloss[[0,1]]}")
    #         print(eval_metrics)

else:

    if args.env_name == "Hopper-v3" or args.env_name == "Walker2d-v3":
        masses = np.linspace(0.5, 4.7, 50)
        frictions = np.linspace(0.4, 2.5, 50)

    else:
        masses = np.linspace(eval_env.sim.model.body_mass[1]/2.0, eval_env.sim.model.body_mass[1]*2.0, 50)
        frictions = np.linspace(eval_env.sim.model.geom_friction[0][0]/2.0, eval_env.sim.model.geom_friction[0][0]*2.0, 50)
    
    gravities = np.linspace(eval_env.sim.model.opt.gravity[2]/2.0, eval_env.sim.model.opt.gravity[2]*2.0, 50)

    # csv_file_path = path+ f"/all_mass_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Mass", "Avg_Return", "Std_Return"])
    #     for mass in masses:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.sim.model.body_mass[1] = mass
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([mass, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for mass {eval_env.sim.model.body_mass[1]}")
    #         print(eval_metrics)

    # eval_env = create_env(args.env_name, args.suite)
    # csv_file_path = path+ f"/all_friction_Adversarial_Diff_avg_returns-{seed}.csv"

    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Friction", "Avg_Return", "Std_Return"])
    #     for friction in frictions:
    #         eval_env = create_env(args.env_name, args.suite)
    #         eval_env.sim.model.geom_friction[0][0] = friction
    #         eval_metrics = evaluate_policy(
    #                     ac.forward_actor,
    #                     eval_env,
    #                     device,
    #                     step,
    #                     dataset,
    #                     use_mean=True,
    #                     n_episodes=10,
    #                     renderer=renderer,
    #                 )
    #         csv_writer.writerow([friction, eval_metrics["avg_return"], eval_metrics["std_return"]])
    #         print(f"Metrics for friction {eval_env.sim.model.geom_friction[0][0]}")
    #         print(eval_metrics)
    csv_file_path = path+ f"/all_gravity_Adversarial_Diff_avg_returns-{seed}.csv"

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Gravity", "Avg_Return", "Std_Return"])
        for gravity in gravities:
            eval_env = create_env(args.env_name, args.suite)
            eval_env.sim.model.opt.gravity[2] = gravity
            eval_metrics = evaluate_policy(
                        ac.forward_actor,
                        eval_env,
                        device,
                        step,
                        dataset,
                        use_mean=True,
                        n_episodes=10,
                        renderer=renderer,
                    )
            csv_writer.writerow([gravity, eval_metrics["avg_return"], eval_metrics["std_return"]])
            print(f"Metrics for gravity {eval_env.sim.model.opt.gravity[2]}")
            print(eval_metrics)

# csv_file_path = path+ f"/all_damping_Adversarial_Diff_avg_returns-{seed}.csv"

# with open(csv_file_path, mode='w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["Damping", "Avg_Return", "Std_Return"])
#     for damping in dampings:
#         eval_env.sim.model.dof_damping[joint] = damping
#         eval_metrics = evaluate_policy(
#                     ac.forward_actor,
#                     eval_env,
#                     device,
#                     step,
#                     dataset,
#                     use_mean=True,
#                     n_episodes=10,
#                     renderer=renderer,
#                 )
#         csv_writer.writerow([damping, eval_metrics["avg_return"], eval_metrics["std_return"]])
#         print(f"Metrics for damping {eval_env.sim.model.dof_damping[joint]}")
#         print(eval_metrics)

print("done")