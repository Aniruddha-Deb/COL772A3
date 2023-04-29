set -e

python3 code/main.py train data/train.jsonl data/dev.jsonl \
--save-t5-name models/t5_prefix_100.pt \
--no-roberta \
--n-prefixes 100

python3 code/main.py test data/devtest.jsonl generations/prefix_100.txt \
--t5-path models/t5_prefix_100.pt \
--roberta-path models/roberta_default.pt

python3 code/evaluations_pp.py data/dev.jsonl generations/prefix_100.txt > results/prefix_100.txt
