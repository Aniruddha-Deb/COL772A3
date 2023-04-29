import sys
import json


def parse(tokens):
    if "(" not in tokens:
        assert ")" not in tokens
        ret = dict()
        start = 0
        mid = 0
        for ii, tok in enumerate(tokens):
            if tok == "«":
                mid = ii
            elif tok == "»":
                key = ' '.join(tokens[start:mid])
                val = ' '.join(tokens[mid + 1:ii])
                ret[key] = val
                start = mid = ii + 1
        return ret

    st = tokens.index("(")
    outer_key = ' '.join(tokens[0:st])
    assert tokens[-1] == ")", " ".join(tokens)

    level = 0
    last = st + 1
    ret = dict()
    for ii in range(st + 1, len(tokens) - 1, 1):
        tok = tokens[ii]
        if tok == "»" and level == 0:
            rr = parse(tokens[last:ii + 1])
            ret.update(rr)
            last = ii + 1
        elif tok == "(":
            level += 1
        elif tok == ")":
            level -= 1
            if level == 0:
                rr = parse(tokens[last:ii + 1])
                ret.update(rr)
                last = ii + 1

    return {outer_key: ret}

def get_slots(sentence):
    toks = sentence.split(' ')
    slots = set()
    skip = False
    for tok in toks[2:-1]:
        if tok == '(' or tok == ')':
            continue
        if tok == "«":
            skip = True
        elif tok == "»":
            skip = False
            continue
        if not skip:
            slots.add(tok)

    return slots

def slot_metrics(gold, pred):
    gold_slots = get_slots(gold)
    pred_slots = get_slots(pred)

    slot_recall = len(gold_slots.intersection(pred_slots))/len(gold_slots) if len(gold_slots) > 0 else 1
    slot_precision = len(gold_slots.intersection(pred_slots))/len(pred_slots) if len(pred_slots) > 0 else 0
    slot_f1 = (2*slot_recall*slot_precision)/(slot_recall + slot_precision) if slot_recall + slot_precision > 0 else 0

    return slot_recall, slot_precision, slot_f1

def load_jsonl(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line.strip()))

    return data


def per_sample_metric(gold, pred):
    ret = dict()
    ret['accuracy'] = int(gold == pred)

    get_intent = lambda x: x.split('(', 1)[0].strip()
    gintent = get_intent(gold)
    pintent = get_intent(pred)
    ret['intent_accuracy'] = int(gintent == pintent)
    ret['slot_recall'], ret['slot_precision'], ret['slot_f1'] = slot_metrics(gold, pred)

    parse_correct = 1
    try:
        _ = parse(pred.split())
    except:
        parse_correct = 0
    ret['parsing_accuracy'] = parse_correct

    return ret


def compute_metrics(data, preds):
    assert len(data) == len(preds), "Different number of samples in data and prediction."

    golds = [x['output'] for x in data]

    metrics = [per_sample_metric(gold, pred) for gold, pred in zip(golds, preds)]
    final_metrics = dict()
    mnames = list(metrics[0].keys())
    for key in mnames:
        final_metrics[key] = sum([met[key] for met in metrics]) / len(golds)
    
    return final_metrics


if __name__ == '__main__':
    data_file = sys.argv[1]
    pred_file = sys.argv[2]

    data = load_jsonl(data_file)
    preds = []
    with open(pred_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            preds.append(line.strip())

    metrics = compute_metrics(data, preds)
    print(json.dumps(metrics, indent=2))
