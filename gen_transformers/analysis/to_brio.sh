#!/bin/bash
echo "Evaluate model with 'cd ~/BRIO/ && bash rank.sh 0 $EXP <brio-rank-exp>'"
exit

export CLASSPATH=~/stanford-corenlp-3.8.0.jar

CAND=16
EXP=$1
FN=$2

if grep -q "cnn" <<< "$EXP"; then
  DATASET="cnn_dailymail"
  BRIO_DATASET="cnndm"
elif grep -q "xsum" <<< "$EXP"; then
  DATASET="xsum"
  BRIO_DATASET="xsum"
elif grep -q "samsum" <<< "$EXP"; then
  DATASET="samsum"
  BRIO_DATASET="samsum"
else
  echo "Dataset not in experiment name"
fi

if grep -q "test" <<< "$FN"; then
  SPLIT="test"
elif grep -q "validation" <<< "$FN"; then
  SPLIT="validation"
elif grep -q "train" <<< "$FN"; then
  SPLIT="train"
else
  echo "Dataset not in experiment name"
fi
echo $DATASET
python convert_to_brio.py --num_candidates $CAND --dataset $DATASET --fn $FN --splits $SPLIT --experiment $EXP
IN_FN="/nlp/projects/faithsum/results/${EXP}/${SPLIT}.txt"
OUT_FN="/nlp/projects/faithsum/results/${EXP}/${SPLIT}.tokenized"
cat $IN_FN | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $OUT_FN
python convert_to_brio.py --num_candidates $CAND --dataset $DATASET --fn $FN --splits $SPLIT --experiment $EXP

FROM_DIR="/nlp/projects/faithsum/results/${EXP}"
TO_DIR="${HOME}/BRIO/${BRIO_DATASET}/"
cp -r $FROM_DIR $TO_DIR
echo "BRIO processed outputs saved to ${TO_DIR}"
echo "Fini!"
echo "Evaluate model with 'cd ~/BRIO/ && bash rank.sh 0 $EXP <brio-rank-exp>'"
