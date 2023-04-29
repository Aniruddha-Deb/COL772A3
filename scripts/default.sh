set -e

python3 code/main.py train data/train.jsonl data/dev.jsonl \
--save-t5-name models/t5_default.pt \
--save-roberta-name models/roberta_default.pt

python3 code/main.py test data/devtest.jsonl generations/default.txt \
--t5-path models/t5_default.pt \
--roberta-path models/roberta_default.pt

python3 code/evaluations.py data/dev.jsonl generations/default.txt > results/default.txt
