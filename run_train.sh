fcosfunc ()
{
    local date=20191015
    local netname=fcos

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --netname $netname --date $date\
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

fcosfunc
