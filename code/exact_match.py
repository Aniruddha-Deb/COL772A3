def matches(y1, y2):
    return ("".join(y1.split()) == "".join(y2.split()))

def exact_match_metric(gold, pred):
    cnt_correct = 0
    for i in range(len(gold)):
        if(matches(gold[i], pred[i])):
            cnt_correct += 1
    return cnt_correct/len(gold)
