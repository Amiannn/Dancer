DECODING_DIR="./dump/2023_05_29__23_12_17"
WORD_FREQ_PATH="./dump/2023_05_29__23_16_34/word_freq.txt"
RARE_WORD_PATH="./blists/aishell/test_1_entities.txt"

./score.sh $DECODING_DIR

python3 -m error_analysis.get_error_word_count \
    "${DECODING_DIR}/result.wrd.txt" \
    $WORD_FREQ_PATH \
    $RARE_WORD_PATH

