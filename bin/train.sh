
DATE=$(date +%H%M%S-%d%m)

for config_path in ./configs/*.jsonnet; do

  CONFIG_NAME=$(basename ${config_path})
  CONFIG_NAME="${CONFIG_NAME%.*}"

  TRAIN_DATA_PATH=./data/train.jsonl \
    VALID_DATA_PATH=./data/valid.jsonl \
    allennlp train ${config_path} -s ./logs/${DATE}/${CONFIG_NAME} --include-package fraud

done