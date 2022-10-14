#r!/bin/bash

start=1
end=5

inter=2
min=1
max=7

for num in $(seq $min $inter $max) ;do
# for avg in $(seq 1 10) ;do
for id in $(seq $start $end) ;do 
#     for video in $(seq 0 21) ;do
#     if [ $((id % 50)) -eq 0 ]; then 
#     id=$(printf %05d $id)
    python3 meta_main.py --save_dir ../exp/meta_test/GOPRO/reblur_exp_attn_restormer_finetune_"$num"_patch_gopro_epoch_"$id"_on_reds \
                   --reset True \
                   --log_file_name test.log \
                   --gpu_id 5 \
                   --random_seed 43 \
                   --num_workers 20 \
                   --dataset REDS \
                   --dataset_dir /disk1/psliu/datasets/benchMark/REDS \
                   --batch_size 1 \
                   --support_optim adam \
                   --optim adam \
                   --img_w 1280 \
                   --img_h 720 \
                   --input_w 1280 \
                   --input_h 720 \
                   --meta True \
                   --meta_test True \
                   --reblur_lr 1e-6 \
                   --deblur_lr 2.5e-6 \
                   --use_inner_lr True \
                   --inner_lr 1e-6 \
                   --meta_lr 1e-7 \
                   --reblur_model attn \
                   --deblur_model restormer \
                   --features 32 \
                   --reblur_result False \
                   --use_reblur_pair True \
                   --reblur_epoch "$id" \
                   --reblur_layers 3 \
                   --reblur_ratio 0.5 \
                   --n_frames 5 \
                   --gan True \
                   --n_critics 1 \
                   --gan_lr 1e-6 \
                   --gan_ratio 0.99 \
                   --reblur_backward False \
                   --cycle_update True \
                   --cycle_block False \
                   --round False \
                   --use_blurrest False \
                   --support_batch 1 \
                   --support_size 256 \
                   --support_epochs "$id" \
                   --query_batch 1 \
                   --task_batch_size 1 \
                   --use_fix_update True \
                   --full_img_sup False \
                   --full_img_exp True \
                   --video_shuffle True \
                   --combine_update False \
                   --n_updates "$num" \
                   --deblur_model_path weights/Restormer/motion_deblurring.pth \
                   --reblur_model_path ../exp/train/GOPRO/reblur_exp_attn_zero2one_gan/model/reblur_01000.pt \
                   --gan_model_path ../exp/train/GOPRO/reblur_exp_attn_zero2one_gan/model/gan_01000.pt \
#                    --reblur_model_path ../exp/meta_train/GOPRO/reblur_exp_attn_restormer_gan_0.99_cycle_update_n_10_patch_e6e6_gopro/model/reblur_"$id".pt \
#                    --deblur_model_path ../exp/meta_train/GOPRO/reblur_exp_attn_restormer_gan_0.99_cycle_update_n_10_patch_e6e6_gopro/model/deblur_"$id".pt \
#     fi
#     done
done
done
