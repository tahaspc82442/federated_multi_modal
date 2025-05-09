# sweep_config.py

import wandb
import pprint # For nicely printing the config
import sys   # To get the python interpreter path

# --- Define the Sweep Configuration ---
# This dictionary contains all the settings for your hyperparameter search.
sweep_config = {
    'method': 'bayes',  # Specifies the search strategy: bayes, random, grid

    'metric': {
      'name': 'eval_unified_test_accuracy', # IMPORTANT: Must match the metric name logged in your training script
      'goal': 'maximize'    # Optimize to maximize this metric (use 'minimize' for loss, etc.)
    },

    'parameters': {

        'TRAINER.MAPLE.LAMBDA_ALIGN': {
            # Option 1: Bayesian or Random Search (define a range)
            'distribution': 'uniform',  # How to sample values (uniform, log_uniform_values, q_uniform, etc.)
            'min': 0.0,                 # Minimum value for lambda_align
            'max': 2.0                  # Maximum value for lambda_align
                                        # (Adjust based on expected scale vs. main loss)

            # Option 2: Grid Search (define specific values)
            # 'values': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        },

        # --- Add other hyperparameters you want to tune here ---
        # Example: Learning Rate (often good to search on a log scale)
        # 'learning_rate': {
        #     'distribution': 'log_uniform_values',
        #     'min': 1e-5,
        #     'max': 1e-2
        # },

        # Example: Batch Size (discrete values)
        # 'batch_size': {
        #     'values': [16, 32, 64]
        # },

        # Example: Optimizer (categorical choice)
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # }

    },

    # --- Define the command to run ---
    # Option 1 (Simpler, less flexible if you have non-hyperparameter args):
    # 'program': 'train.py', # Just specify the script name

    # Option 2 (Recommended for your case): Use the 'command' key
    'command': [
        '${env}',                       # Sets environment variables (like WANDB_RUN_ID)
        f'{sys.executable}',            # Use the current Python interpreter
        'train.py',                     # Your training script (can replace '${program}' if 'program' key is also set)

        # --- FIXED ARGUMENTS ---
        # Add all arguments that DON'T change between sweep runs
        '--root', '/raid/biplab/taha',
        '--seed', '3243', # Note: If you want to sweep seed, move it to 'parameters'
        '--trainer', 'MaPLeFederated',
        '--dataset-config-file', 'configs/datasets/PatternNet.yaml',
        '--config-file', 'configs/trainers/MaPLeFederated/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml',
        '--output-dir', 'output/PatternNet/MaPLeFederated/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/seed3243', # Consider making parts of this dynamic if seed changes
        'DATASET.NUM_SHOTS', '16', # Make sure your script parses this type of argument correctly

        # --- SWEPT ARGUMENTS ---
        # '${args}' will be replaced by wandb agent with '--param_name=value'
        # for each hyperparameter defined in the 'parameters' section above.
        '${args}'
    ]
}

# --- Optional: Initialize the sweep directly from this script ---
# This part is executed only when you run `python sweep_config.py`
if __name__ == '__main__':
    # --- REPLACE THESE PLACEHOLDERS ---
    WANDB_PROJECT_NAME = "my_fed_project"  # Replace with your W&B project name
    WANDB_ENTITY_NAME = "mohd-taha82442-iit-bombay" # Replace with your W&B username or team name
    # --- END OF PLACEHOLDERS ---

    print("--- Sweep Configuration ---")
    pprint.pprint(sweep_config)
    print("-------------------------")

    print(f"\nInitializing sweep for project='{WANDB_PROJECT_NAME}', entity='{WANDB_ENTITY_NAME}'...")

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=WANDB_PROJECT_NAME,
                           entity=WANDB_ENTITY_NAME)

    print(f"\nSweep successfully initialized!")
    print(f"Sweep ID: {sweep_id}")
    print("\nTo run the sweep agent, use the following command:")
    # CORRECTED AGENT COMMAND EXAMPLE:
    print(f"wandb agent {WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}")
    print("\nNOTE: The agent will automatically run the command specified in the 'command' section of the sweep configuration.")
    print("Remember to set CUDA_VISIBLE_DEVICES before running the agent, e.g.:")
    print(f"CUDA_VISIBLE_DEVICES=0 wandb agent {WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}")