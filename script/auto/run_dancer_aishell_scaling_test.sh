# input datas
TRANSCRIPTION_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"
TRANSCRIPTION_NBEST_PATH="./datas/aishell_test_set/asr_transcription/conformer/nbest"
MANSCRIPTION_PATH="./datas/aishell_test_set/ref"

# entity datas
ENTITY_ROOT="./datas/entities/aishell/scales"
ENTITY_TEST_PATH="./datas/entities/aishell/test/test_1_entities.txt"
ENTITY_FILES=("all_1000_scale_entities.txt" "all_2000_scale_entities.txt" "all_3000_scale_entities.txt" "all_4000_scale_entities.txt" "all_5000_scale_entities.txt" "all_6000_scale_entities.txt" "all_7000_scale_entities.txt" "all_8000_scale_entities.txt" "all_9000_scale_entities.txt" "all_10000_scale_entities.txt" "all_11000_scale_entities.txt" "all_12000_scale_entities.txt" "all_13000_scale_entities.txt" "all_14000_scale_entities.txt" "all_15000_scale_entities.txt")
ENTITY_CONTENT_PATH="./datas/entities/aishell/descriptions/ctx.json"
ENTITY_VECTORS_PATH="./datas/entities/aishell/descriptions/embeds.npy"

# detection model
DETECTION_MODEL_TYPE="bert_detector"
DETECTION_MODEL_PATH="./ckpts/ner/best_model"

# retrieval model
RETRIEVAL_MODEL_TYPE="prsr_retriever"
RETRIEVAL_MODEL_PATH="./ckpts/ranker/dpr_biencoder.39"

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
        --retrieval_model_path $RETRIEVAL_MODEL_PATH             \
        --detection_model_type $DETECTION_MODEL_TYPE             \
        --detection_model_path $DETECTION_MODEL_PATH             \
        --use_rejection $USE_REJECTION                           \
        --asr_nbest_transcription_path $TRANSCRIPTION_NBEST_PATH \
        --entity_path $ENTITY_PATH                               \
        --entity_test_path $ENTITY_TEST_PATH                     \
        --entity_content_path $ENTITY_CONTENT_PATH               \
        --entity_vectors_path $ENTITY_VECTORS_PATH
    echo '________________________________________________________'
done