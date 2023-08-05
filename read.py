import os
import pandas as pd
import numpy as np
from document_processor import *
from guess_class import *

#get path
#return a matrix form of all the documents
class Folder_Reader:
    def __init__ (self, path = '/Users/minhhanh/Documents/Study/NYU/Y3/Summer/PA/HW3/Text_Mining/', data_path = 'dataset_3/data/'):
        self.path = path
        self.data_path = data_path
        self.texts = []
        self.labels = []

    def read_folder (self):

        all_df = pd.DataFrame()

        folder_list = os.listdir(self.data_path)
        for folder in folder_list:
            if folder == '.DS_Store':
                continue

            file_list = os.listdir(self.data_path+folder)
            #print(file_list)

            for file in file_list:
                self.labels.append(folder)
                
                with open(self.data_path+folder+'/'+file) as f:
                    lines = f.readlines()

                text = ''
                for line in lines:
                    text += line

                self.texts.append(text)

                doc_processor = Document_Processor(file, folder)
                doc_vector = doc_processor.process_pipeline(text)
                #print(doc_vector)
                all_df = pd.concat([all_df,doc_vector])
        
        #all_df.to_csv('before_idf.csv')

        
        ### MODIFICATION FOR HW4
        # process unknown 
        unknown_files = os.listdir('unknown')
        for file in unknown_files:
            print(file)
            with open('unknown/'+file) as f:
                lines = f.readlines()

            text = ''
            for line in lines:
                text += line

            self.texts.append(text)

            doc_processor = Document_Processor(file, 'unknown')
            doc_vector = doc_processor.process_pipeline(text)
            print(doc_vector)
            doc_vector.to_csv('doc_vec.csv')
            all_df = pd.concat([all_df,doc_vector])
            print(all_df.shape)
        

        #TF-IDF
        number_of_docs = 34#len(self.labels)
        term_doc_num = number_of_docs - all_df.isna().sum()


        idf = np.log(number_of_docs/term_doc_num)

        for ind in idf.index:
            all_df[ind] = all_df[ind] * idf[ind] 

        #drop columns 
        #words that only appear in 1 document
        to_keep = term_doc_num[term_doc_num > 1].index
        all_df = all_df[to_keep]

        '''
        classes = guess_topics (all_df)
        with open('folder_topics.txt','w') as f: 
            for key, value in classes.items(): 
                f.write('%s:%s\n' % (key, value))

        '''

        all_df.fillna(value=0,inplace=True)

        return all_df
