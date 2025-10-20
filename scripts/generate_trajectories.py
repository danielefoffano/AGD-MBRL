import polygrad.utils as utils
import torch
import numpy as np
from polygrad.utils.envs import create_env
from tqdm import tqdm
import pickle
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    config: str = "config.simple_maze"
    seed: int = 1
    run_number: int = 0
    n_trajectories: int = 100000
    max_path_length: int = 1000
    checkpoint_path: str = None  # Path to saved model checkpoint
    output_dir: str = "./generated_trajectories"
    use_mean: bool = True  # Use mean of policy distribution instead of sampling
    save_batch_size: int = 1000  # Save trajectories in batches


args = Parser().parse_args()

print(f"Generating {args.n_trajectories} trajectories")
print(f"Seed: {args.seed}")
utils.set_all_seeds(args.seed)

# Create environment
env = create_env(args.env_name, args.suite)

random_episodes = utils.rl.random_exploration(args.n_prefill_steps, env)

# Load configs
configs = utils.create_configs(args, env)
if configs["render_config"] is not None:
    renderer = configs["render_config"]()
else:
    renderer = None

# Initialize models
model = configs["model_config"]()
diffusion = configs["diffusion_config"](model)
dataset = configs["dataset_config"](random_episodes)  # Empty dataset for normalization
ac = configs["ac_config"](normalizer=dataset.normalizer)

# Load checkpoint if provided
if args.checkpoint_path is not None:
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ac.load_state_dict(checkpoint['actor_critic'])
    if 'normalizer' in checkpoint:
        dataset.normalizer = checkpoint['normalizer']
    print("Checkpoint loaded successfully")

ac.eval()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------------------------------------------------------#
# --------------------------- generate trajectories ---------------------------#
# -----------------------------------------------------------------------------#

def generate_trajectory():
    """Generate a single trajectory by rolling out the policy."""
    state, _ = env.reset()
    done = False
    t = 0
    
    trajectory = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "sim_states": [],
    }
    
    while not done and t < args.max_path_length:
        # Get action from policy
        with torch.no_grad():
            policy_dist = ac.forward_actor(
                torch.from_numpy(state).float().to(device), 
                normed_input=False
            )
            if args.use_mean:
                act = policy_dist.mean.cpu().numpy()
            else:
                act = policy_dist.sample().cpu().numpy()
        
        # Step environment
        next_state, rew, term, trunc, info = env.step(act)
        done = term or trunc
        t += 1
        
        # Store transition
        trajectory["observations"].append(state.copy())
        trajectory["actions"].append(act.copy())
        trajectory["next_observations"].append(next_state.copy())
        trajectory["rewards"].append(rew)
        trajectory["terminals"].append(term)
        trajectory["timeouts"].append(trunc)
        
        if "sim_state" in info.keys():
            trajectory["sim_states"].append(info["sim_state"].copy())
        else:
            trajectory["sim_states"].append(None)
        
        state = next_state
    
    # Convert to numpy arrays
    trajectory = {key: np.array(trajectory[key]) for key in trajectory.keys()}
    
    return trajectory


# Generate trajectories
all_trajectories = []
returns = []
lengths = []

print("Generating trajectories...")
for i in tqdm(range(args.n_trajectories)):
    traj = generate_trajectory()
    all_trajectories.append(traj)
    
    ret = np.sum(traj["rewards"])
    returns.append(ret)
    lengths.append(len(traj["rewards"]))
    
    # Save in batches to avoid memory issues
    if (i + 1) % args.save_batch_size == 0:
        batch_file = os.path.join(
            args.output_dir, 
            f"trajectories_batch_{(i + 1) // args.save_batch_size}.pkl"
        )
        with open(batch_file, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f"\nSaved batch {(i + 1) // args.save_batch_size}")
        all_trajectories = []  # Clear memory

# Save remaining trajectories
if len(all_trajectories) > 0:
    batch_file = os.path.join(
        args.output_dir,
        f"trajectories_batch_final.pkl"
    )
    with open(batch_file, 'wb') as f:
        pickle.dump(all_trajectories, f)

# Save statistics
stats = {
    "mean_return": np.mean(returns),
    "std_return": np.std(returns),
    "min_return": np.min(returns),
    "max_return": np.max(returns),
    "mean_length": np.mean(lengths),
    "std_length": np.std(lengths),
    "n_trajectories": args.n_trajectories,
}

stats_file = os.path.join(args.output_dir, "statistics.pkl")
with open(stats_file, 'wb') as f:
    pickle.dump(stats, f)

print("\n" + "="*50)
print("Trajectory Generation Complete")
print("="*50)
print(f"Total trajectories: {args.n_trajectories}")
print(f"Mean return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
print(f"Return range: [{stats['min_return']:.2f}, {stats['max_return']:.2f}]")
print(f"Mean length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
print(f"Saved to: {args.output_dir}")
print("="*50)