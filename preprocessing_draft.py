import stanza
import re

def get_stopwords ():
    stopwords = list()
    with open('stopwords.txt') as f:
        stopwords = f.readline().replace('"', '')
        stopwords = stopwords.split(', ')

    return stopwords

def remove_stopwords (sentence):
    sentence = sentence.lower()
    stopwords = get_stopwords()
    sentence_arr = sentence.split()

    filtered = []

    for word in sentence_arr:
        stop = False
        for i, stopword in enumerate(stopwords):
            if stopword == word:
                stop = True
        if stop == False: 
            filtered.append(word)

    return ' '.join(filtered)



def filter_stopwords (text_array):
    #input: array of words in text
    #parse through all words -> remove if in stopwords list

    stopwords = get_stopwords()
    sort_text = sorted(text_array)

    stop_pointer = 0
    filtered_text = []

    for i, word in enumerate(sort_text):
        while word > stopwords[stop_pointer]:
            stop_pointer += 1  

        if word != stopwords[stop_pointer]:
            filtered_text.append(word)


    return filtered_text
    
#stanza.download('en','/Users/minhhanh/Documents/Study/NYU/Y3/Summer/PA/HW3/python_nlp/')

def preprocess (text):  
    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
    tok = tokenizer(text)

    lemmatizing = stanza.Pipeline(lang='en', processors='tokenize,lemma', tokenize_pretokenized=True)
    lem = lemmatizing(tok)

def sliding (text):
    MAX_WINDOW_SIZE = 4
    frequencies = dict()
    for i, word in enumerate (text):
        for j in range(2,MAX_WINDOW_SIZE+1):
            if i+j > len(text):
                continue
            phrase = ' '.join(text[i:i+j])
            
            if phrase in frequencies.keys():
                frequencies[phrase] += 1
            
            else:
                frequencies[phrase] = 1
    
    doc_length = len(text)

    top_phrase = dict()
    for key in frequencies.keys():
        if frequencies[key] >= 3:
            top_phrase[key] = frequencies[key]
    #print(frequencies)
    return top_phrase

# pipeline
#1. tokenization
#2. lemmatization
#3. NER (maybe before removing stop words?)
#5. sliding windows
#6. Remove stop words (in each phrase and overall)
#4. matrix


with open('dataset_3/data/C1/article01.txt') as f:
    lines = f.readlines()

    text_arr = list()

    text = ''
    for line in lines:
        line = re.sub('[^A-Za-z0-9 ]+',' ', line)
        text += line
        text_arr = text_arr + line.lower().split()



    filter_text = remove_stopwords(text)
    #print(filter_text)
    top_phrase = sliding(filter_text.split())
    print(top_phrase)


    #print(get_stopwords())

    filtered_text = filter_stopwords(text_arr)
    #tokenize (filtered_text)

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner')
    doc = nlp(text)
    lemma = [word.lemma for sent in doc.sentences for word in sent.words]
    print('lemma')
    print(lemma)

    ent = [ent.text for sent in doc.sentences for ent in sent.ents]
    print('entity')
    print(ent)
    #print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')

    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
    tok = tokenizer(text)

    lemmatizing = stanza.Pipeline(lang='en', processors='tokenize,lemma', tokenize_pretokenized=True)
    lem = lemmatizing(tok)
    #print(lem)

    print('len lemma', len(lemma))
    print('len doc', len(text_arr))

t = 'In an unprecedented move in local commercial aviation'
#print(remove_stopwords (t))


#print(lemmatizing(doc))




