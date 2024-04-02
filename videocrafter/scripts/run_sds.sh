ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"
export CUDA_VISIBLE_DEVICES=1

name="cfg10_lr1_t20"
python scripts/evaluation/videocrafter2_utils.py \
    --seed 123 \
    --mode 'base' \
    --ckpt_path $ckpt \
    --config $config \
    --savedir $res_dir/$name \
    --n_samples 1 \
    --bs 1 --height 320 --width 512 \
    --prompt_file $prompt_file \
    --cfg 10.0 --lr 0.1 --cfg_temporal 20.0 \
