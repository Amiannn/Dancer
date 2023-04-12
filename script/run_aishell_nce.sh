TRANSCRIPTION_PATH="/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/text"
TRANSCRIPTION_NBEST_PATH="/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir"
DETECTION_MODEL_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
ENTITY_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_21_02__16_09_32/test_1_entities.json"

python3 -m entity_correction                                 \
    --asr_transcription_path $TRANSCRIPTION_PATH             \
    --asr_nbest_transcription_path $TRANSCRIPTION_NBEST_PATH \
    --detection_model_path $DETECTION_MODEL_PATH             \
    --entity_path $ENTITY_PATH                               \