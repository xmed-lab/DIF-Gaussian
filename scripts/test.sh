gpu=2

n_view=6
dst_name=LUNA16
name=DIFG-${n_view}v
n_epoch=350

CUDA_VISIBLE_DEVICES=$gpu python code/evaluate.py \
    --name $name \
    --epoch $n_epoch \
    --dst_name $dst_name \
    --split test \
    --num_views $n_view \
    --out_res_scale 1.0
