

import wandb
import pprint # For nicely printing the config
import sys   # To get the python interpreter path

# --- Define the Sweep Configuration ---
# This dictionary contains all the settings for your hyperparameter search.
sweep_config = {
    'method': 'grid',  # Specifies the search strategy: grid

    'metric': {
      'name': 'eval_unified_test_accuracy', # IMPORTANT: Must match the metric name logged in your training script
      'goal': 'maximize'    # Optimize to maximize this metric (use 'minimize' for loss, etc.)
    },

    'parameters': {
        # --- Parameters for DualPrompt Trainer ---
        'TRAINER.DUALPROMPT.N_CTX': {
            # Define the specific values you want to try for N_CTX in the grid
            # Example: 'values': [1, 2, 4, 8]
            'values': [2]  # <<<--- REPLACE WITH YOUR LIST OF N_CTX VALUES
        },

        'TRAINER.DUALPROMPT.PROMPT_DEPTH': {
            # Define the specific values you want to try for PROMPT_DEPTH in the grid
            # Example: 'values': [8, 11, 16, 24]
            'values': [3,4,5,6,7,8,9]  # <<<--- REPLACE WITH YOUR LIST OF PROMPT_DEPTH VALUES
        },

        'TRAINER.DUALPROMPT.LAMBDA_ALIGN': {
            # Define the specific values you want to try for LAMBDA_ALIGN in the grid
            # Example based on your initial value and wanting to try others:
            # 'values': [0.0, 0.01, 0.1, 0.5, 1.0]
            'values': [0.1,0.3,0.5,0.7,0.9]  # <<<--- REPLACE WITH YOUR LIST OF LAMBDA_ALIGN VALUES
        },

        # --- Add other hyperparameters you want to tune here if any ---
        # Example: Learning Rate (if you were also grid searching this)
        # 'learning_rate': {
        #     'values': [1e-5, 5e-5, 1e-4]
        # },

        # Example: Batch Size (discrete values)
        # 'batch_size': {
        #     'values': [16, 32]
        # },
    },

    
    'command': [
        '${env}',                       # Sets environment variables (like WANDB_RUN_ID)
        f'{sys.executable}',            # Use the current Python interpreter
        'train.py',                     # Your training script

        # --- FIXED ARGUMENTS ---
        # Add all arguments that DON'T change between sweep runs
        '--root', '/raid/biplab/taha',
        '--seed', '6000', # Note: If you want to sweep seed, move it to 'parameters'
        '--trainer', 'DualPromptFL', # This seems to be the base trainer name
                                       # The swept DualPrompt parameters will be passed as --TRAINER.DUALPROMPT.XYZ
        '--dataset-config-file', 'configs/datasets/PatternNet.yaml',
        '--config-file', 'configs/trainers/DualPromptFL/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml',
        # Consider making parts of output-dir dynamic if other parameters also change,
        # or rely on W&B to group runs by sweep.
        '--output-dir', 'output/PatternNet/DualPromptFL/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots_sweep/seed3243', # Changed to a sweep-specific subdir
        'DATASET.NUM_SHOTS', '16',

        # --- SWEPT ARGUMENTS ---
        # '${args}' will be replaced by wandb agent with arguments like:
        #   --TRAINER.DUALPROMPT.N_CTX=value_from_grid
        #   --TRAINER.DUALPROMPT.PROMPT_DEPTH=value_from_grid
        #   --TRAINER.DUALPROMPT.LAMBDA_ALIGN=value_from_grid
        # Your train.py script needs to be able to parse these arguments.
        # Many config systems like yacs or OmegaConf handle dot-separated paths automatically.
        '${args}'
    ]
}

# --- Optional: Initialize the sweep directly from this script ---
# This part is executed only when you run `python sweep_config.py`
if __name__ == '__main__':
    # --- REPLACE THESE PLACEHOLDERS ---
    WANDB_PROJECT_NAME = "my_fed_project"  # Replace with your W&B project name
    WANDB_ENTITY_NAME = "mohd-taha82442-iit-bombay" # Replace with your W&B username or team name
   
    if not sweep_config['parameters']['TRAINER.DUALPROMPT.N_CTX']['values'] or \
       not sweep_config['parameters']['TRAINER.DUALPROMPT.PROMPT_DEPTH']['values'] or \
       not sweep_config['parameters']['TRAINER.DUALPROMPT.LAMBDA_ALIGN']['values']:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Grid values for N_CTX, PROMPT_DEPTH, or LAMBDA_ALIGN are  !!!")
        print("!!!          still empty in the sweep_config. Please edit the script   !!!")
        print("!!!          and fill them in before running.                           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # You might want to exit here if running directly and values are not set
        # sys.exit("Exiting: Please define grid values in the script.")


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
    print(f"wandb agent {WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}")
    print("\nNOTE: The agent will automatically run the command specified in the 'command' section of the sweep configuration.")
    print("Remember to set CUDA_VISIBLE_DEVICES if needed before running the agent, e.g.:")
    print(f"CUDA_VISIBLE_DEVICES=0 wandb agent {WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}/{sweep_id}")