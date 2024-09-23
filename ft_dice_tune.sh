
for lr in 2e-4 1e-4 7e-5 6e-5
do
for wd in 1e-2
do
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python ft.py \
    --lr=$lr \
    --wd=$wd \
    --bs=512 \
    --ep=15 \
    --losstype=dice \
    --report_name=dice_lr${lr}_wd${wd}
done
done