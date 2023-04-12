TRANSCRIPTION_PATH="/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_asr/asr1/exp/asr_whisper_tiny_finetune_lr1e-5_adamw_wd1e-2_3epochs/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/cv+fsw+ii+esun/test_esun2022/text"
TRANSCRIPTION_NBEST_PATH="/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_asr/asr1/exp/asr_whisper_tiny_finetune_lr1e-5_adamw_wd1e-2_3epochs/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/cv+fsw+ii+esun/test_esun2022/logdir"
DETECTION_MODEL_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
ENTITY_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_30_03__02_07_00/good_entity.json"

python3 -m entity_correction                                 \
    --asr_transcription_path $TRANSCRIPTION_PATH             \
    --asr_nbest_transcription_path $TRANSCRIPTION_NBEST_PATH \
    --detection_model_path $DETECTION_MODEL_PATH             \
    --entity_path $ENTITY_PATH                               \