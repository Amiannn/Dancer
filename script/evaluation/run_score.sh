ENTITY_PATH="./datas/entities/aishell/test/test_1_entities.txt"
REF_PATH="./datas/aishell_test_set/asr_transcription/conformer/confuse/ref_homophone_set"
HYP_PATH="./datas/aishell_test_set/asr_transcription/conformer/confuse/hyp_homophone_set"

python3 -m error_analysis.score \
    --entity_path $ENTITY_PATH \
    --ref_path $REF_PATH \
    --hyp_path $HYP_PATH