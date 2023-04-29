set -e

python3 code/main.py train data/train.jsonl data/dev.jsonl \
--save-t5-name models/t5_prefix_20.pt \
--no-roberta \
--n-prefixes 20

python3 code/main.py test data/devtest.jsonl generations/prefix_20.txt \
--t5-path models/t5_prefix_20.pt \
--roberta-path models/roberta_default.pt

python3 code/evaluations.py data/dev.jsonl generations/prefix_20.txt > results/prefix_20.txt
