CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python ft.py \
    --lr=5e-5 \
    --wd=1e-3 \
    --bs=512 \
    --ep=10 \
    --losstype=ce \
    --report_name=test