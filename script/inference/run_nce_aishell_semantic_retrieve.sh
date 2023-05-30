TRANSCRIPTION_PATH="/share/nas165/amian/experiments/speech/espnet_old/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/text"
TRANSCRIPTION_NBEST_PATH="/share/nas165/amian/experiments/speech/espnet_old/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir"
MANSCRIPTION_PATH="/share/nas165/amian/experiments/speech/espnet_old/workspace/esun_zh_tcpgen/asr1/data/aishell_ner/test/text"
DETECTION_MODEL_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
ENTITY_PATH="/share/nas165/amian/experiments/speech/EntityCorrector/blists/aishell/test_1_entities.txt"

# semantic retrieval
RETRIEVAL_MODEL_PATH="/share/nas165/amian/experiments/nlp/DPR/outputs/2023-03-07/16-45-46/output/dpr_biencoder.39"
ENTITY_CONTENT_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_02_27__15_58_45_test/aishell_ner_ctx.json"
ENTITY_VECTORS_PATH="/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_15_03__14_36_47/embeds.npy"

DETECTION_MODEL_TYPE="bert_detector"
RETRIEVAL_MODEL_TYPE="semantic_retriever"
USE_REJECTION="True"

python3 -m entity_correction                                 \
    --asr_transcription_path $TRANSCRIPTION_PATH             \
    --asr_manuscript_path $MANSCRIPTION_PATH                 \
    --retrieval_model_type $RETRIEVAL_MODEL_TYPE             \
    --retrieval_model_path $RETRIEVAL_MODEL_PATH             \
    --detection_model_type $DETECTION_MODEL_TYPE             \
    --detection_model_path $DETECTION_MODEL_PATH             \
    --use_rejection $USE_REJECTION                           \
    --asr_nbest_transcription_path $TRANSCRIPTION_NBEST_PATH \
    --entity_path $ENTITY_PATH                               \
    --entity_content_path $ENTITY_CONTENT_PATH               \
    --entity_vectors_path $ENTITY_VECTORS_PATH