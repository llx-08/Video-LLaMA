nohup python train.py --cfg-path  ./train_configs/audiobranch_stage2_finetune.yaml > debug.log 2>&1 &

#nohup torchrun --nproc_per_node=2 train.py --cfg-path  ./train_configs/audiobranch_stage2_finetune.yaml
