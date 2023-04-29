set -e

python3 code/main.py train data/train.jsonl data/dev.jsonl \
--no-roberta \
--save-t5-name models/t5_small_lr.pt \
--t5-lr 1e-5

python3 code/main.py test data/devtest.jsonl generations/small_lr.txt \
--t5-path models/t5_default.pt \
--roberta-path models/roberta_default.pt

python3 code/evaluations.py data/dev.jsonl generations/small_lr.txt > results/small_lr.txt
