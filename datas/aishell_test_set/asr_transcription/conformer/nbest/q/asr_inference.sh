#!/bin/bash
cd /share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.asr_inference --batch_size 1 --ngpu 0 --data_path_and_name_and_type dump/raw/aishell_ner/test/wav.scp,speech,kaldi_ark --key_file exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/keys.${SGE_TASK_ID}.scp --asr_train_config exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/config.yaml --asr_model_file exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/valid.acc.ave_10best.pth --output_dir exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/output.${SGE_TASK_ID} --config conf/decode_asr_transformer.yaml 
EOF
) >exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/asr_inference.$SGE_TASK_ID.log
time1=`date +"%s"`
 ( python3 -m espnet2.bin.asr_inference --batch_size 1 --ngpu 0 --data_path_and_name_and_type dump/raw/aishell_ner/test/wav.scp,speech,kaldi_ark --key_file exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/keys.${SGE_TASK_ID}.scp --asr_train_config exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/config.yaml --asr_model_file exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/valid.acc.ave_10best.pth --output_dir exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/output.${SGE_TASK_ID} --config conf/decode_asr_transformer.yaml  ) 2>>exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/asr_inference.$SGE_TASK_ID.log >>exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/asr_inference.$SGE_TASK_ID.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/asr_inference.$SGE_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/asr_inference.$SGE_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/q/sync/done.151660.$SGE_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/q/asr_inference.log   -t 1:32 /share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/q/asr_inference.sh >>exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir/q/asr_inference.log 2>&1
