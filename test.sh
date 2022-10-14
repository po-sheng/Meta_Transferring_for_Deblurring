### testing
python3 main.py --save_dir ../exp/test/GOPRO/reblur_exp_attn_zero2one_gan \
               --reset True \
               --log_file_name test.log \
               --test True \
               --save_results True \
               --reblur_result False \
               --random_seed 43 \
               --gpu_id 9 \
               --num_workers 4 \
               --dataset GOPRO \
               --dataset_dir /disk1/psliu/datasets/benchMark/GOPRO_Large \
               --img_h 720 \
               --img_w 1280 \
               --input_h 720 \
               --input_w 1280 \
               --reblur_model attn \
               --deblur_model mprnet \
               --features 32 \
               --reblur_layers 3 \
               --n_frames 5 \
               --full_img_exp True \
               --deblur_model_path weights/MPRNet/model_deblurring.pth \
#                --reblur_model_path ../exp/train/GOPRO/reblur_exp_attn_mprnet_gan_1000/model/reblur_00100.pt \

