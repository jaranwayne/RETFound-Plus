
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --savemodel \
    --batch_size 16 \
    --world_size 1 \
    --epochs 50 \
    --blr 2e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path /data/home/wangzheyuan/cipher/data/RetiFOUNDPlus/clean_htn_data \
    --input_size 224 \
    --task train_htn_incidence_using_rfp \
    --finetune /data/home/wangzheyuan/cipher/misc/checkpoints/ibot/retfound_plus_student_encoder.pth
