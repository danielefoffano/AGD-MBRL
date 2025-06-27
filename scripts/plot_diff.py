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
                mass_cart_file_pattern = f"all_mass_cart_{method}_avg_returns-"
                mass_pole_file_pattern = f"all_mass_pole_{method}_avg_returns-"
                frictionloss_file_pattern = f"all_frictionloss_{method}_avg_returns-"
                ac_gears_file_pattern = f"all_ac_gears_{method}_avg_returns-"
                ac_gear_file_pattern = f"all_ac_gear_{method}_avg_returns-"
                
                mass_file = None
                friction_file = None
                gravity_file = None
                mass_cart_file = None
                mass_pole_file = None
                frictionloss_file = None
                ac_gears_file = None
                
                for file in os.listdir(seed_path):
                    if file.startswith(mass_file_pattern) and file.endswith(".csv"):
                        mass_file = os.path.join(seed_path, file)
                    elif file.startswith(friction_file_pattern) and file.endswith(".csv"):
                        friction_file = os.path.join(seed_path, file)
                    elif file.startswith(gravity_file_pattern) and file.endswith(".csv"):
                        gravity_file = os.path.join(seed_path, file)
                    elif file.startswith(mass_cart_file_pattern) and file.endswith(".csv"):
                        mass_cart_file = os.path.join(seed_path, file)
                    elif file.startswith(mass_pole_file_pattern) and file.endswith(".csv"):
                        mass_pole_file = os.path.join(seed_path, file)
                    elif file.startswith(frictionloss_file_pattern) and file.endswith(".csv"):
                        frictionloss_file = os.path.join(seed_path, file)
                    elif (file.startswith(ac_gears_file_pattern) or file.startswith(ac_gear_file_pattern)) and file.endswith(".csv"):
                        ac_gears_file = os.path.join(seed_path, file)
                        
                    
                if mass_file and os.path.exists(mass_file):
                    mass_df = pd.read_csv(mass_file, usecols=["Mass", "Avg_Return"])
                    mass_df.rename(columns={"Mass": "Value"}, inplace=True)
                    mass_df["Type"] = "Mass"
                    mass_df["Method"] = method
                    mass_df["Environment"] = environment
                    data.append(mass_df)
                
                if friction_file and os.path.exists(friction_file):
                    friction_df = pd.read_csv(friction_file, usecols=["Friction", "Avg_Return"])
                    friction_df.rename(columns={"Friction": "Value"}, inplace=True)
                    friction_df["Type"] = "Friction"
                    friction_df["Method"] = method
                    friction_df["Environment"] = environment
                    data.append(friction_df)

                if gravity_file and os.path.exists(gravity_file):
                    gravity_df = pd.read_csv(gravity_file, usecols=["Gravity", "Avg_Return"])
                    gravity_df.rename(columns={"Gravity": "Value"}, inplace=True)
                    gravity_df["Type"] = "Gravity"
                    gravity_df["Method"] = method
                    gravity_df["Environment"] = environment
                    data.append(gravity_df)

                if frictionloss_file and os.path.exists(frictionloss_file):
                    frictionloss_df = pd.read_csv(frictionloss_file, usecols=["Frictionloss", "Avg_Return"])
                    frictionloss_df.rename(columns={"Frictionloss": "Value"}, inplace=True)
                    frictionloss_df["Type"] = "Friction"
                    frictionloss_df["Method"] = method
                    frictionloss_df["Environment"] = environment
                    data.append(frictionloss_df)

                if mass_cart_file and os.path.exists(mass_cart_file):
                    mass_cart_df = pd.read_csv(mass_cart_file, usecols=["Mass", "Avg_Return"])
                    mass_cart_df.rename(columns={"Mass": "Value"}, inplace=True)
                    mass_cart_df["Type"] = "Mass Cart"
                    mass_cart_df["Method"] = method
                    mass_cart_df["Environment"] = environment
                    data.append(mass_cart_df)

                if mass_pole_file and os.path.exists(mass_pole_file):
                    mass_pole_df = pd.read_csv(mass_pole_file, usecols=["Mass", "Avg_Return"])
                    mass_pole_df.rename(columns={"Mass": "Value"}, inplace=True)
                    mass_pole_df["Type"] = "Mass Pole"
                    mass_pole_df["Method"] = method
                    mass_pole_df["Environment"] = environment
                    data.append(mass_pole_df)
                
                if ac_gears_file and os.path.exists(ac_gears_file):
                    ac_gear_df = pd.read_csv(ac_gears_file, usecols=["ac_gears", "Avg_Return"])
                    ac_gear_df.rename(columns={"ac_gears": "Value"}, inplace=True)
                    ac_gear_df["Type"] = "Actuator gear"
                    ac_gear_df["Method"] = method
                    ac_gear_df["Environment"] = environment
                    data.append(ac_gear_df)
    
    if data:
        result_df = pd.concat(data, ignore_index=True)
        agg_df = result_df.groupby(["Type", "Method", "Environment", "Value"], as_index=False).agg(
            Avg_Return_Mean=("Avg_Return", "mean"),
            Std_Dev=("Avg_Return", "std"),
            Sample_Size=("Avg_Return", "count")
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
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
    
    # For easier labeling
    x_label = f"{data_type} relative change"
    y_label = "Average Return"
    
    # Plot group 1 on axes[0]
    ax_left = axes#[0]
    subset_left = env_df[env_df["Method"].isin(group1_methods)]
    
    for method, group in subset_left.groupby("Method"):
        group = group.copy()
        group["Ratio"] = group["Value"] / nominal_values[environment][data_type]

        # Filter the rows to only include 0.6 <= Ratio <= 1.2
        if environment == "Hopper" and data_type == "Mass":
            group = group[(group["Ratio"] >= 0.3) & (group["Ratio"] <= np.inf)]
        
        if environment == "Hopper" and data_type == "Friction":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 2.25)]

        if environment == "Walker" and data_type == "Mass":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= np.inf)]

        if environment == "Walker" and data_type == "Friction":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3.0)]

        if environment == "Walker" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.6) & (group["Ratio"] <= 1.4)]
        
        if environment == "Hopper" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.15)]

        if environment == "Cheetah" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]
        
        if environment == "InvertedPendulum" and data_type == "Gravity":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 1.5)]

        if environment == "InvertedPendulum" and data_type == "Mass Pole":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3)]

        if environment == "InvertedPendulum" and data_type == "Mass Cart":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= 2)]

        if environment == "Reacher" and data_type == "Frictionloss":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 50)]

        if environment == "Reacher" and data_type == "Actuator gear":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= 2)]

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
    #ax_left.set_ylabel(y_label)
    
    # Plot group 2 on axes[1]
    ax_right = axes#[1]
    subset_right = env_df[env_df["Method"].isin(group2_methods)]
    
    for method, group in subset_right.groupby("Method"):
        group = group.copy()
        group["Ratio"] = group["Value"] / nominal_values[environment][data_type]

        # Filter the rows to only include 0.6 <= Ratio <= 1.2
        if environment == "Hopper" and data_type == "Mass":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= np.inf)]

        if environment == "Hopper" and data_type == "Friction":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 2.25)]

        if environment == "Walker" and data_type == "Mass":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= np.inf)]

        if environment == "Walker" and data_type == "Friction":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3.0)]

        if environment == "Walker" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.6) & (group["Ratio"] <= 1.4)]
        
        if environment == "Hopper" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.15)]

        if environment == "Cheetah" and data_type == "Gravity":
            group = group[(group["Ratio"] >= 0.8) & (group["Ratio"] <= 1.4)]

        if environment == "InvertedPendulum" and data_type == "Gravity":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 1.5)]

        if environment == "InvertedPendulum" and data_type == "Mass Pole":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 3)]

        if environment == "InvertedPendulum" and data_type == "Mass Cart":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= 2)]

        if environment == "Reacher" and data_type == "Frictionloss":
            group = group[(group["Ratio"] >= -np.inf) & (group["Ratio"] <= 50)]

        if environment == "Reacher" and data_type == "Actuator gear":
            group = group[(group["Ratio"] >= 0.5) & (group["Ratio"] <= 2)]

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
    
    # fig.legend(
    #     final_handles,
    #     final_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.56, -0.2),  # adjust to your preference
    #     ncol=3,
    #     handlelength=1,
    #     columnspacing=1.0,
    #     handletextpad=0.2
    # )
    
    plt.tight_layout()
    plt.savefig(f"{environment}_{data_type}_final_new.pdf", bbox_inches="tight")
    plt.close(fig)

# ======================
# Example usage
# ======================
if __name__ == "__main__":
    # Directories & data
    logs_directory = "./scripts/logs"
    environments_list = ["Cheetah"] #["Cheetah", "Hopper", "Walker"]#["InvertedPendulum"]#["Cheetah", "Hopper", "Walker", "InvertedPendulum", "Reacher"]
    # Choose which data type to plot
    data_type = "Gravity"
    
    # Two different sets of methods
    methods_list_group_1 = ["Adversarial_Diff", "DR_PPO", "M2TD3"]
    methods_list_group_2 = ["Adversarial_Diff", "PPO", "TRPO", "Polygrad"]
    
    # Nominal values for reference
    nominal_values = {
        "Cheetah": {"Mass": 6.36, "Friction": 0.4, "Gravity": -9.81},
        "Hopper":  {"Mass": 3.53, "Friction": 1.0, "Gravity": -9.81},
        "Walker":  {"Mass": 3.53, "Friction": 0.7, "Gravity": -9.81},
        "InvertedPendulum":{"Mass Cart": 10.47197551, "Mass Pole": 5.01859164, "Gravity": -9.81},
        "Reacher":{"Friction": 1.0, "Actuator gear": 200}
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