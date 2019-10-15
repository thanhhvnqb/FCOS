keypointrcnnfunc ()
{
    local date=20190902
    local netname=keypointrcnn
    python tools/train_net.py --netname $netname --date $date\
            --config-file configs/mpprcnn/e2e_keypoint_rcnn_R_50_FPN_1x.yaml 2>&1 | tee -a "./out/run_${date}_$netname.log"

}

keypointrcnnfunc
