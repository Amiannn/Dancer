# input datas
TRANSCRIPTION_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"
TRANSCRIPTION_NBEST_PATH="./datas/aishell_test_set/asr_transcription/conformer/nbest"
MANSCRIPTION_PATH="./datas/aishell_test_set/ref"

# entity datas
ENTITY_ROOT="./datas/entities/aishell"
ENTITY_TEST_PATH="./datas/entities/aishell/test/test_1_entities.txt"
ENTITY_FILES=("all_0_entities.txt" "all_0.02_entities.txt" "all_0.05_entities.txt" "all_0.1_entities.txt" "all_0.2_entities.txt")

# detection model
DETECTION_MODEL_TYPE="bert_detector"
DETECTION_MODEL_PATH="./ckpts/ner/best_model"

# retrieval model
RETRIEVAL_MODEL_TYPE="pinyin_retriever"

# rejection
USE_REJECTION="True"

for filename in ${ENTITY_FILES[@]}
do
    ENTITY_PATH=${ENTITY_ROOT}/${filename}
    echo 'Coverage test: '${ENTITY_PATH}

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