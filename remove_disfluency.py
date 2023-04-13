disfluent_words = ['uh', 'uhh', 'um', 'umm', 'uhm','uhmm', 'oh', 'ah', 'er', 'ummm', 'err']

def trim(s):
    s = s.lower()
    if s[-1] == ',':
        s = s[0:-1]
    return s

def remove_disfluency(ds):
    for i in range(len(ds.data)):
        if ds.data[i]['pattern'] != 'disfluency':
            continue
        words_list = ds.data[i]['input'].split(' ')
        last_word = ''
        s_last_word = ''
        for j in range(len(words_list)):
            if(len(words_list[j]) == 0):
                continue
            word = trim(words_list[j])
            if(word in disfluent_words):
                words_list[j] = ''
            elif (word == last_word):
                words_list[j] = ''
            elif (word == s_last_word):
                words_list[j] = ''
            else:
                s_last_word = last_word
                last_word = word
        words_list = list(filter(lambda x: x != '', words_list))        
        ds.data[i]['input'] = " ".join(words_list)
    return ds