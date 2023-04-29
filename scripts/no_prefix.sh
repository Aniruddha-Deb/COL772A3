set -e

python3 code/main.py train data/train.jsonl data/dev.jsonl \
--save-t5-name models/t5_no_prefix_r.pt \
--no-roberta \
--n-prefixes 0

python3 code/main.py test data/devtest.jsonl generations/no_prefix.txt \
--t5-path models/t5_no_prefix_r.pt \
--roberta-path models/roberta_default.pt

python3 code/evaluations_pp.py data/dev.jsonl generations/no_prefix.txt > results/no_prefix.txt
