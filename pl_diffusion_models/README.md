# PL Prediction Models v0.1.0

# Change Log
## Version 0.1.0
- first release version

## Supported Dataset Version
- Unified Dataset v0.2.0
    - Waymo motion dataset 1.2.1
    - Argoverse2 motion dataset
    - NuScenes prediction dataset


## Supported Model Version
- LitMultiTFModel v0.1.0

## How to launch training process
`python launch.py [self_defined_args] fit --config ./config/cli_config/config.yaml`
- supported self_defined_args:
    - `--debug`: True or False, enable debug mode
    - `--compute_metrics`: True or False, enable metrics computation in fit process
    - `--use_wandb`: True or False, enable wandb logging
    - `--use_earlystop`: True or False, enable early stopping


python launch.py --debug True --compute_metrics False --use_wandb False --use_earlystop False

## How to launch testing process
`python launch.py [self_defined_args] test --config ./config/cli_config/config.yaml --ckpt_path ./PATH_TO_YOUR_CHECKPOINT`
- supported self_defined_args:
    - `--debug`: True or False, enable debug mode


## Future works
- [x] fast dev run
- [x] tuning for lr, batch_size, etc.
