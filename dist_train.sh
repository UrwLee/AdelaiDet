export NCCL_SOCKET_IFNAME=eth0
python tools/train_net.py \
	--machine-rank $1 \
	--num-machines 2 \
	--dist-url "tcp://$2:$3" \
	--config-file experiments/MS_R_50_1x.yaml \
       	--num-gpus 4
