TRAIN_PATH="./datas/ced/aishell_trainset_plus_conformer_nbest10_trainset_decode_result.json"
MODEL_TYPE="bert-base-chinese"
WANDB="DANCER_CED_EXP"

EPOCH=10
BATCH=256

CUDA_VISIBLE_DEVICES=0 python3 -m train_ced \
    --train_path $TRAIN_PATH                \
    --model_type $MODEL_TYPE                \
    --wandb ${WANDB}                        \
    --epoch $EPOCH                          \
    --batch $BATCH                          