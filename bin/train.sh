
DATE=$(date +%H%M%S-%d%m)

for config_path in ./configs/*.jsonnet; do

  CONFIG_NAME=$(basename ${config_path})
  CONFIG_NAME="${CONFIG_NAME%.*}"

  echo ">>>>>> training ${CONFIG_NAME}"

  TRAIN_DATA_PATH=./data_full/train.jsonl \
    VALID_DATA_PATH=./data_full/valid.jsonl \
    allennlp train ${config_path} -s ./logs/${DATE}/${CONFIG_NAME} --include-package fraud

done


for config_path in ./configs/*.jsonnet; do

  CONFIG_NAME=$(basename ${config_path})
  CONFIG_NAME="${CONFIG_NAME%.*}"

  echo ">>>>>> training ${CONFIG_NAME}"

  TRAIN_DATA_PATH=./data_shuffled_train/train.jsonl \
    VALID_DATA_PATH=./data_shuffled_train/valid.jsonl \
    allennlp train ${config_path} -s ./logs/${DATE}/${CONFIG_NAME} --include-package fraud

done


for config_path in ./configs/*.jsonnet; do

  CONFIG_NAME=$(basename ${config_path})
  CONFIG_NAME="${CONFIG_NAME%.*}"

  echo ">>>>>> training ${CONFIG_NAME}"

  TRAIN_DATA_PATH=./data_shuffled_all/train.jsonl \
    VALID_DATA_PATH=./data_shuffled_all/valid.jsonl \
    allennlp train ${config_path} -s ./logs/${DATE}/${CONFIG_NAME} --include-package fraud

done