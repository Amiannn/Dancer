ENTITY_PATH="./datas/entities/aishell/all_ctx_entities.txt"
ENTITY_TEST_ROOT="./datas/entities/aishell/shots"
ENTITY_TEST=("test_0_shot_entities.txt" "test_1_shot_entities.txt" "test_5_shot_entities.txt" "test_10_shot_entities.txt" "test_20_shot_entities.txt" "test_100_shot_entities.txt")

REF_PATH="./datas/aishell_test_set/ref"
HYP_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"

for test_entity in ${ENTITY_TEST[@]}
do
    ENTITY_TEST_PATH=${ENTITY_TEST_ROOT}/${test_entity}
    echo 'Unseen test: '${ENTITY_TEST_PATH}

    python3 -m error_analysis.score \
        --entity_path $ENTITY_TEST_PATH \
        --ref_path $REF_PATH \
        --hyp_path $HYP_PATH

    echo '________________________________________________________'
done