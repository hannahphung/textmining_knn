import stanza
import re
import os
import pandas as pd

#get path
#return a matrix form of all the documents
class Document_Processor:
    def __init__ (self, doc_name, doc_folder, path = '/Users/minhhanh/Documents/Study/NYU/Y3/Summer/PA/HW3/Text_Mining/', data_path = 'dataset_3/data/'):
        self.doc_name = doc_name
        self.doc_folder = doc_folder
        self.path = path
        self.data_path = data_path
       

    def stanza_lemma_ner (self, text, downloaded = False):

        if downloaded == False:
            stanza.download('en',self.path)

        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner')
        doc = nlp(text)

        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        ners = [ent.text for sent in doc.sentences for ent in sent.ents]

        long_ners = []
        for ner in ners:
            if len(ner.split()) > 1:
                long_ners.append(ner)
        
        #return a list of all lemma (root word)
        #return a list of entity with more than 1 word (ner)
        return lemmas, long_ners
    
    def sliding_window (self, text):
        text = re.sub('[^A-Za-z0-9 ]+',' ', text)
        text = self.remove_stopwords_text(text)

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
        
        #doc_length = len(text)

        top_phrase = dict()
        for key in frequencies.keys():
            if frequencies[key] >= 3:
                top_phrase[key] = frequencies[key]
        #print(frequencies)
        return top_phrase
    
    def get_stopwords (self):
        stopwords = list()
        with open('stopwords.txt') as f:
            stopwords = f.readline().replace('"', '')
            stopwords = stopwords.split(', ')

        return stopwords

    def filter_stopwords (self, text_array):

        #input: array of words in text
        #parse through all words -> remove if in stopwords list

        stopwords = self.get_stopwords()
        sort_text = sorted(text_array)

        stop_pointer = 0
        filtered_text = []

        for i, word in enumerate(sort_text):
            while word > stopwords[stop_pointer]:
                stop_pointer += 1  

            if word != stopwords[stop_pointer]:
                filtered_text.append(word)


        return filtered_text
    
    def remove_stopwords_text (self,sentence):
        sentence = sentence.lower()
        stopwords = self.get_stopwords()
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
    
    def process_pipeline (self, text):
        #1. stanza tokenizer, lemma, and ner
        #2. ngrams: sliding windows with frequency >= 3, check if in ner list
        #3. remove stop words in ner and ngrams
        #4. remove stop words
        #5. frequency matrix 
        #6. merge ner and end grams
        #7. TF-IDF: TF in this part and IDF when doing all documents

        #1. stanza tokenizer, lemma, and ner
        lemmas, long_ners = self.stanza_lemma_ner (text)
        #print(lemmas)

        #2. ngrams: sliding windows with frequency >= 3, check if in ner list
        ngrams = self.sliding_window (text)

        ngrams = list(map(str.lower,ngrams))
        long_ners = list(map(str.lower,long_ners))

        filtered_ngrams = []
        for gram in ngrams:
            if gram not in long_ners:
                filtered_ngrams.append(gram)

        #3. remove stop words in ner and ngrams
        final_ner_ngrams = []
        for ner in long_ners:
            final_ner_ngrams.append(self.remove_stopwords_text(ner))

        for gram in filtered_ngrams:
            final_ner_ngrams.append(self.remove_stopwords_text(gram))

        #4. remove stop words
        filtered_lemmas = self.filter_stopwords (lemmas)

        #5. frequency matrix 
        frequency_dict = dict()
        for lemma in filtered_lemmas:
            lemma = lemma.lower()
            #special characters 
            if lemma.isalpha():
                if lemma in frequency_dict.keys():
                    frequency_dict[lemma] += 1

                else:
                    frequency_dict[lemma] = 1
        if '' in frequency_dict.keys():
            frequency_dict.pop('')
        if ' ' in frequency_dict.keys():
            frequency_dict.pop(' ')
        
        length = len (list(frequency_dict.keys()))

        for word in final_ner_ngrams:
            #add to frequency dict
            if word.isalpha():
                if word in frequency_dict.keys():
                    frequency_dict[word] += 1

                else:
                    frequency_dict[word] = 1

            #remove frequency of components
            components = word.split()
            for comp in components:
                if comp in frequency_dict.keys():
                    frequency_dict[comp] -= 1


        ind = self.doc_folder + '-' + self.doc_name
        df = pd.DataFrame(frequency_dict, index = [ind])

        #df.to_csv('og_term_vector.csv')

        #7. TF
        #length = len(frequency_dict.keys())
        df = df / length

        #return a vector that rep the document
        return df

'''
with open('dataset_3/data/C7/article01.txt') as f:
    lines = f.readlines()
    text = ''
    for line in lines:
        text += line

processor = Document_Processor('article1', 'C7')
processor.process_pipeline(text)
'''
