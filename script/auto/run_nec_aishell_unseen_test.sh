# input datas
TRANSCRIPTION_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"
TRANSCRIPTION_NBEST_PATH="./datas/aishell_test_set/asr_transcription/conformer/nbest"
MANSCRIPTION_PATH="./datas/aishell_test_set/ref"

# entity datas
ENTITY_PATH="./datas/entities/aishell/all_ctx_entities.txt"
ENTITY_TEST_ROOT="./datas/entities/aishell/shots"
ENTITY_TEST=("test_0_shot_entities.txt" "test_1_shot_entities.txt" "test_5_shot_entities.txt" "test_10_shot_entities.txt" "test_20_shot_entities.txt" "test_100_shot_entities.txt")

# detection model
DETECTION_MODEL_TYPE="bert_detector"
DETECTION_MODEL_PATH="./ckpts/ner/best_model"

# retrieval model
RETRIEVAL_MODEL_TYPE="pinyin_retriever"

# rejection
USE_REJECTION="True"

for test_entity in ${ENTITY_TEST[@]}
do
    ENTITY_TEST_PATH=${ENTITY_TEST_ROOT}/${test_entity}
    echo 'Unseen test: '${ENTITY_TEST_PATH}

    python3 -m entity_correction                                 \
        --asr_transcription_path $TRANSCRIPTION_PATH             \
        --asr_manuscript_path $MANSCRIPTION_PATH                 \
        --retrieval_model_type $RETRIEVAL_MODEL_TYPE             \
        --detection_model_type $DETECTION_MODEL_TYPE             \
        --detection_model_path $DETECTION_MODEL_PATH             \
        --use_rejection $USE_REJECTION                           \
        --asr_nbest_transcription_path $TRANSCRIPTION_NBEST_PATH \
        --entity_path $ENTITY_PATH                               \
        --entity_test_path $ENTITY_TEST_PATH
    echo '________________________________________________________'
done