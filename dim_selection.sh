#!/usr/bin/env bash

# https://github.com/ziyin-dl/word-embedding-dimensionality-selection
# will save results to experiments folder
MAIN_SCRIPT_PATH="/Users/fursovia/Documents/word-embedding-dimensionality-selection"
DATA_PATH="/Users/fursovia/Desktop/fraud/data/all_treatments.txt"

LSA_CONFIG_PATH="/Users/fursovia/Desktop/fraud/configs/fraud_lsa.yml"
GLOVE_CONFIG_PATH="/Users/fursovia/Desktop/fraud/configs/fraud_glove.yml"
WORD2VEC_CONFIG_PATH="/Users/fursovia/Desktop/fraud/configs/fraud_lsa.yml"


if [[ ! -d "$MAIN_SCRIPT_PATH" ]];
then
    echo "Cloning the repo..."
    mkdir $MAIN_SCRIPT_PATH
    git clone https://github.com/ziyin-dl/word-embedding-dimensionality-selection $MAIN_SCRIPT_PATH
fi

cd $MAIN_SCRIPT_PATH && python -m main --file $DATA_PATH --config_file $LSA_CONFIG_PATH --algorithm lsa
cd $MAIN_SCRIPT_PATH && python -m main --file $DATA_PATH --config_file $GLOVE_CONFIG_PATH --algorithm glove
cd $MAIN_SCRIPT_PATH && python -m main --file $DATA_PATH --config_file $WORD2VEC_CONFIG_PATH --algorithm word2vec

cp -r $MAIN_SCRIPT_PATH/params/* experiments/