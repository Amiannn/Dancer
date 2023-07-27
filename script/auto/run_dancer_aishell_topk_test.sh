# input datas
TRANSCRIPTION_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"
TRANSCRIPTION_NBEST_PATH="./datas/aishell_test_set/asr_transcription/conformer/nbest"
MANSCRIPTION_PATH="./datas/aishell_test_set/ref"

# entity datas
ENTITY_PATH="./datas/entities/aishell/all_ctx_entities.txt"
ENTITY_TEST_PATH="./datas/entities/aishell/test/test_1_entities.txt"
ENTITY_CONTENT_PATH="./datas/entities/aishell/descriptions/ctx.json"
ENTITY_VECTORS_PATH="./datas/entities/aishell/descriptions/embeds.npy"

# detection model
DETECTION_MODEL_TYPE="bert_detector"
DETECTION_MODEL_PATH="./ckpts/ner/best_model"

# retrieval model
RETRIEVAL_MODEL_TYPE="prsr_retriever"
RETRIEVAL_MODEL_PATH="./ckpts/ranker/dpr_biencoder.39"

PRSR_TOPK=("1" "5" "10" "20" "40" "80" "100")

# rejection
USE_REJECTION="True"

for topk in ${PRSR_TOPK[@]}
do
    echo 'TopK testing: '${topk}
    
    python3 -m entity_correction                                 \
        --asr_transcription_path $TRANSCRIPTION_PATH             \
        --asr_manuscript_path $MANSCRIPTION_PATH                 \
        --retrieval_model_type $RETRIEVAL_MODEL_TYPE             \
        --retrieval_model_path $RETRIEVAL_MODEL_PATH             \
        --prsr_topk $topk                                        \
        --prsr_prsr 0.5                                          \
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