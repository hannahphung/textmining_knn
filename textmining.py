from read import *
from document_processor import *
from cluster import *
from sklearn.decomposition import PCA
from visualizer import *
from guess_class import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class TextMining:
    def __init__ (self):
        self.df = None
        self.predicted_label = None
        self.modified_predicted = None

    def process (self):
        Preprocessor = Folder_Reader() 
        self.df = Preprocessor.read_folder()   
        self.df.to_csv('word_matrix_with_unknown.csv')  

    def cluster (self):
        self.process()

        pca = PCA(n_components=10)
        reduced_data = pd.DataFrame(pca.fit_transform(self.df))

        clusterer = Cluster (reduced_data)
        self.predicted_label, centroids = clusterer.Kmeans(K=3, maxiter=100, similarity = 'cosine')
        
        return self.predicted_label, centroids
    
    def visualize (self, actual_label):
        self.cluster()
        self.modified_predicted = majority_class (self.predicted_label, actual_label)
        print('Predicted',self.modified_predicted)
        print('Actual',actual_label)
        visualizer = Visualizer(self.df,self.modified_predicted, actual_label)
        visualizer.plot()

    def evaluate (self, actual_label):
        prec = precision_score(y_true=actual_label, y_pred=self.predicted_label, average="micro",
                                         zero_division=0)
        print('Precision score:', prec)
        rec = recall_score(y_true=actual_label, y_pred=self.modified_predicted, average="micro")
        print('Recall score:', rec)
        f1 = f1_score(y_true=actual_label, y_pred=self.modified_predicted, average="micro")
        print('F1 score:', f1)

        conf_matrix = confusion_matrix(y_true=actual_label,y_pred=self.modified_predicted)
        print('Confusion matrix')
        print(pd.DataFrame(conf_matrix))
        
        



