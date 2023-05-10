import wandb
log_dir = r"C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\DP\run1501"
wandb.tensorboard.patch(root_logdir=log_dir)
run = wandb.init()
pass