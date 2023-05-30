CUDA_VISIBLE_DEVICES=1 python3 -m error_analysis.units.eval_detector \
    --entity_path "./blists/aishell/test_1_entities.txt" \
    --ref_path "/share/nas165/amian/experiments/speech/espnet_old/workspace/esun_zh_tcpgen/asr1/data/aishell_ner/test/text" \
    --hyp_path "/share/nas165/amian/experiments/speech/espnet_old/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/text" \
    --model_type "bert_detector" \
    --model_path "/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"