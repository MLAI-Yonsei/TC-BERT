fgamma=1.0
for lr in 2e-4 1e-4 7e-5 6e-5
do
for wd in 1e-2
do
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python ft.py \
    --lr=$lr \
    --wd=$wd \
    --bs=512 \
    --focal_gamma=$fgamma \
    --ep=15 \
    --losstype=focal \
    --report_name=focal_lr${lr}_wd${wd}_gam${fgamma}
done
done