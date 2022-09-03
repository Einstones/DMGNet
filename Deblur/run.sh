#CUDA_VISIBLE_DEVICES=0 python -u train.py \
#                     --sidd_path=/home/share/rongyao/sidd/SIDD_Small_sRGB_Only \
#                     --batch_size=4 \
#                     --patch_size=128 \
#                     --nEpochs=3000 \
#                     --workers=4

CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --master_port 50002 --nproc_per_node=2  train.py 
