import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.titlesize'] = 16   # fontsize of the axes title
plt.rcParams['axes.labelsize'] = 25   # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'] = 20  # fontsize of the tick labels
plt.rcParams['ytick.labelsize'] = 20  # fontsize of the tick labels
plt.rcParams['legend.fontsize'] = 25  # legend fontsize
plt.rcParams['figure.titlesize'] = 16 # figure title fontsize

def extract_data(logs_dir, methods, environments):
    data = []
    
    for method in methods:
        for environment in environments:
            base_path = os.path.join(logs_dir, f"{method}_{environment}")
            if not os.path.exists(base_path):
                print(f"Skipping {base_path}, does not exist.")
                continue
            
            for seed_dir in os.listdir(base_path):
                if not seed_dir.startswith("seed"):
                    continue
                
                seed_path = os.path.join(base_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                
                # Extracting Mass and Avg_Return
                mass_file_pattern = f"all_mass_{method}_avg_returns-"
                friction_file_pattern = f"all_friction_{method}_avg_returns-"
                gravity_file_pattern = f"all_gravity_{method}_avg_returns-"
                
                mass_file = None
                friction_file = None
                gravity_file = None
                
                for file in os.listdir(seed_path):
                    if ("training" in file.lower()) and file.endswith(".csv"):
                        training_file = os.path.join(seed_path, file)
                    # elif file.startswith(friction_file_pattern) and file.endswith(".csv"):
                    #     friction_file = os.path.join(seed_path, file)
                    # elif file.startswith(gravity_file_pattern) and file.endswith(".csv"):
                    #     gravity_file = os.path.join(seed_path, file)
                    
                if training_file and os.path.exists(training_file):
                    training_df = pd.read_csv(training_file, usecols=["step", "avg_return"])
                    training_df.rename(columns={"step": "Value"}, inplace=True)
                    training_df["Type"] = "Training"
                    training_df["Method"] = method
                    training_df["Environment"] = environment
                    data.append(training_df)
                
                # if friction_file and os.path.exists(friction_file):
                #     friction_df = pd.read_csv(friction_file, usecols=["Friction", "Avg_Return"])
                #     friction_df.rename(columns={"Friction": "Value"}, inplace=True)
                #     friction_df["Type"] = "Friction"
                #     friction_df["Method"] = method
                #     friction_df["Environment"] = environment
                #     data.append(friction_df)

                # if gravity_file and os.path.exists(gravity_file):
                #     gravity_df = pd.read_csv(gravity_file, usecols=["Gravity", "Avg_Return"])
                #     gravity_df.rename(columns={"Gravity": "Value"}, inplace=True)
                #     gravity_df["Type"] = "Gravity"
                #     gravity_df["Method"] = method
                #     gravity_df["Environment"] = environment
                #     data.append(gravity_df)
    
    if data:
        result_df = pd.concat(data, ignore_index=True)
        agg_df = result_df.groupby(["Type", "Method", "Environment", "Value"], as_index=False).agg(
            Avg_Return_Mean=("avg_return", "mean"),
            Std_Dev=("avg_return", "std"),
            Sample_Size=("avg_return", "count")
        )
        
        # Compute standard error (SEM)
        agg_df["Std_Error"] = agg_df["Std_Dev"] / np.sqrt(agg_df["Sample_Size"])
        
        # Define confidence intervals using SEM
        agg_df["CI_Lower"] = agg_df["Avg_Return_Mean"] - agg_df["Std_Error"]
        agg_df["CI_Upper"] = agg_df["Avg_Return_Mean"] + agg_df["Std_Error"]
        
        return agg_df
    else:
        print("No relevant data found.")
        return None

def plot_two_groups_side_by_side(
    df, 
    data_type, 
    environment, 
    group1_methods, 
    group2_methods,
    nominal_values,
    method_display_names,
    method_colors
):
    """
    This function creates a single figure with two subplots:
    - Left subplot for methods in group1_methods
    - Right subplot for methods in group2_methods
    Both share the Y-axis and a single legend.
    """
    
    # Filter df to the given environment and data_type
    env_df = df[(df["Environment"] == environment) & (df["Type"] == data_type)]
    
    # If there's no data, skip
    if env_df.empty:
        print(f"No data for {environment} - {data_type}")
        return
    
    # Create the figure with 2 subplots, side by side
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 6), sharey=True)
    
    # For easier labeling
    x_label = "Steps"
    y_label = "Average Return"
    
    # Plot group 1 on axes[0]
    ax_left = axes#[0]
    subset_left = env_df[env_df["Method"].isin(group1_methods)]
    
    for method, group in subset_left.groupby("Method"):
        group = group.copy()
        group["Ratio"] = group["Value"] #/ nominal_values[environment][data_type]

        # Filter the rows to only include 0.6 <= Ratio <= 1.2
        # if environment == "Hopper" and data_type == "Mass":
        #     group = group[(group["Ratio"] >= 0.3) & (group["Ratio"] <= np.inf)]

        # if environment == "Walker" and data_type == "Mass":
        #     group = group[(group["Ratio"] >= 0.3) & (group["Ratio"] <= np.inf)]

        # if environment == "Walker" and data_type == "Friction":
        #     group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3.0)]

        # if environment == "Walker" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.6) & (group["Ratio"] <= 1.4)]
        
        # if environment == "Hopper" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]

        # if environment == "Cheetah" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]

        ax_left.plot(
            group["Ratio"],
            group["Avg_Return_Mean"],
            label=method_display_names.get(method, method),
            color=method_colors.get(method, "black"),
            linewidth=3
        )
        ax_left.fill_between(
            group["Ratio"],
            group["CI_Lower"],
            group["CI_Upper"],
            alpha=0.2,
            color=method_colors.get(method, "black")
        )
        ax_left.set_xlim(group["Ratio"].min(), group["Ratio"].max())
    
    #ax_left.set_title(f"{environment} - Group 1")
    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel(y_label)
    
    # Plot group 2 on axes[1]
    ax_right = axes#[1]
    subset_right = env_df[env_df["Method"].isin(group2_methods)]
    
    for method, group in subset_right.groupby("Method"):
        group = group.copy()
        group["Ratio"] = group["Value"] #/ nominal_values[environment][data_type]

        # Filter the rows to only include 0.6 <= Ratio <= 1.2
        # if environment == "Hopper" and data_type == "Mass":
        #     group = group[(group["Ratio"] >= 0.3) & (group["Ratio"] <= np.inf)]

        # if environment == "Walker" and data_type == "Mass":
        #     group = group[(group["Ratio"] >= 0.3) & (group["Ratio"] <= np.inf)]

        # if environment == "Walker" and data_type == "Friction":
        #     group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3.0)]

        # if environment == "Walker" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.6) & (group["Ratio"] <= 1.4)]
        
        # if environment == "Hopper" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]

        # if environment == "Cheetah" and data_type == "Gravity":
        #     group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]

        ax_right.plot(
            group["Ratio"],
            group["Avg_Return_Mean"],
            label=method_display_names.get(method, method),
            color=method_colors.get(method, "black"),
            linewidth=3 
        )
        ax_right.fill_between(
            group["Ratio"],
            group["CI_Lower"],
            group["CI_Upper"],
            alpha=0.2,
            color=method_colors.get(method, "black")
        )
        ax_right.set_xlim(group["Ratio"].min(), group["Ratio"].max())
    
    #ax_right.set_title(f"{environment} - Group 2")
    ax_right.set_xlabel(x_label)
    # No need for a Y-label here because sharey=True
    # --- Create a single legend for the entire figure ---
    # We grab all handles and labels from both subplots
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    
    # Combine them while preserving order, but removing duplicates
    all_handles = handles_left + handles_right
    all_labels = labels_left + labels_right

    # If you want to remove duplicates in a simpler way:
    unique_pairs = list(dict(zip(all_labels, all_handles)).items())
    final_labels, final_handles = zip(*unique_pairs)
    
    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),  # adjust to your preference
        ncol=6,
        handlelength=1,
        columnspacing=1.0,
        handletextpad=0.2
    )
    #fig.suptitle(f"{environment}", fontsize=25)
    plt.tight_layout()
    plt.savefig(f"{environment}_{data_type}_final_new.pdf", bbox_inches="tight")
    plt.close(fig)

# ======================
# Example usage
# ======================
if __name__ == "__main__":
    # Directories & data
    logs_directory = "./scripts/logs"
    environments_list = ["Cheetah", "Hopper", "Walker", "InvertedPendulum", "Reacher"]
    # Choose which data type to plot
    data_type = "Training"
    
    # Two different sets of methods
    methods_list_group_1 = ["Adversarial_Diff", "DR_PPO", "M2TD3"]
    methods_list_group_2 = ["Adversarial_Diff", "PPO", "TRPO", "Polygrad"]
    
    # Nominal values for reference
    nominal_values = {
        "Cheetah": {"Mass": 6.36, "Friction": 0.4, "Gravity": -9.81},
        "Hopper":  {"Mass": 3.53, "Friction": 1.0, "Gravity": -9.81},
        "Walker":  {"Mass": 3.53, "Friction": 0.7, "Gravity": -9.81}
    }
    
    # Friendly display names for legend
    method_display_names = {
        "Adversarial_Diff": "AD-RRL",
        "DR_PPO": "DR-PPO",
        "M2TD3": "M2TD3",
        "PPO": "PPO",
        "TRPO": "TRPO",
        "Polygrad": "PolyGRAD"
    }
    
    # Method-specific colors
    method_colors = {
        "Adversarial_Diff": "blue",
        "DR_PPO": "brown",
        "M2TD3": "purple",
        "PPO": "red",
        "TRPO": "orange",
        "Polygrad": "green"
    }

    # Extract the data for the *union* of both groups
    all_methods = list(set(methods_list_group_1 + methods_list_group_2))
    df = extract_data(logs_directory, all_methods, environments_list)

    if df is not None and not df.empty:
        # Create a side-by-side plot for each environment
        for env in environments_list:
            plot_two_groups_side_by_side(
                df=df,
                data_type=data_type,
                environment=env,
                group1_methods=methods_list_group_1,
                group2_methods=methods_list_group_2,
                nominal_values=nominal_values,
                method_display_names=method_display_names,
                method_colors=method_colors
            )
    else:
        print("Dataframe is None or empty. No plots created.")