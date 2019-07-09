import os
import nltk

for path, subdirs, files in os.walk('/home/ebringas/ml/ham/beck-s/'):
    for x in files:
        print(os.path.join(path, name))
        try:
            with open(os.path.join(path,name), mode= 'r', errors='ignore') as f:
                mail_list = f.read().split('From')
                tokens = []
                print(len(mail_list)) 
                for i, x in enumerate(mail_list):
                    print(i+1)
                    raw_message = parse_raw_message('from:' + x.lower()).get('body')
                    if raw_message is not None:
                        untagged = nltk.word_tokenize(raw_message)
                        tagged = nltk.pos_tag(untagged)
                        print('tagged')
                        for term, pos in tagged:
                            if len(term) > 2 and pos in ['JJ', 'NN',"NNP"]:
                                tokens.append(term)
        except:
            pass
print(len(tokens))
freq = nltk.FreqDist(tokens)
freq.plot(100, cumulative=False)

