EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='2'

NUM_GPU=4
python train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--text_encoder '/../../plm/bert-base-uncased' \
--checkpoint '/../../plm/TCL_4M.pth' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--model_save_epoch 1 > dgm4_train_${EXPID}.log 2>&1