import wandb
run = wandb.init()
artifact = run.use_artifact('mohd-taha82442-iit-bombay/my_fed_project/aggregator_checkpoint:v115', type='model')
artifact_dir = artifact.download()