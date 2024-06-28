gpus=6,7
cfg_path=./configs/default.yaml

n_view=6
dst_name=LUNA16
name=DIFG-${n_view}v

mkdir -p ./logs/$name

CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
    --master_port 2025 \
    --nproc_per_node 2 \
    code/train.py \
        --name $name \
        --batch_size 4 \
        --epoch 400 \
        --dst_name $dst_name \
        --num_views $n_view \
        --random_view \
        --cfg_path $cfg_path \
        --dist \
        >> ./logs/$name/train.log 2>&1 &
