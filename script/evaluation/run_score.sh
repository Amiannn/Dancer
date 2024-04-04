ENTITY_PATH="./datas/entities/aishell/test/test_1_entities.txt"

REF_PATH="./datas/aishell_test_set/ref"
HYP_PATH="./datas/aishell_test_set/asr_transcription/conformer/hyp"

python3 -m error_analysis.score \
    --entity_path $ENTITY_PATH \
    --ref_path $REF_PATH \
    --hyp_path $HYP_PATH