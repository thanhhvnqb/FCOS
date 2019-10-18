fcosfunc ()
{
    local date=20191017
    local netname=fcos
    # local netname=fcos_loss

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --netname $netname --date $date\
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

rpnfunc ()
{
    local date=20191017
    local netname=rpn
    # local netname=rpn_iouloss

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --netname $netname --date $date\
    --config-file configs/rpn_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

maskrcnnfunc ()
{
    local date=20191017
    local netname=maskrcnn_iouloss
    # local netname=fcos_loss

    python -m torch.distributed.launch\
    --nproc_per_node=2\
    --master_port=$((RANDOM + 10000))\
    tools/train_net.py --netname $netname --date $date\
    --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml\
    DATALOADER.NUM_WORKERS 2\
    OUTPUT_DIR ./out/ 2>&1 | tee -a "./out/run_${date}_$netname.log"
}

rpnfunc
# maskrcnnfunc
# fcosfunc
