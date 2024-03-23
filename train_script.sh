#nohup python train.py --cfg-path  ./train_configs/audiobranch_stage2_finetune.yaml

nohup torchrun --nproc_per_node=2 train.py --cfg-path  ./train_configs/audiobranch_stage2_finetune.yaml
