OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file experiments/MS_R_50_1x_7.yaml \
    --eval-only \
    --num-gpus 4 \
    OUTPUT_DIR /opt/tiger/toutiao/labcv/dmx_loop/data/visiual/condinst/7 \
    MODEL.WEIGHTS /opt/tiger/toutiao/labcv/dmx_loop/checkpoints/condinst/MS_R_50_1x_7/model_0059999.pth