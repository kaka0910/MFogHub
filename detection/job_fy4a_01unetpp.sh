# ----------------------01unetpp SGD 1e-4----------------------------
CUDA_VISIBLE_DEVICES=0 python /groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/code/train_fy4a.py \
            --dataset fy4a --batch-size 16 --workers 1 \
            --model aspp_unet --checkname aspp_unet \
            --model_optimizer SGD \
            --lr 0.0001 \
            --epochs 100 \
            --weight-decay 0.05 \
            --model_savefolder /groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/FY4A_output_cs0506/01unetpp/ \
