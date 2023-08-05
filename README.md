# Descriptive Modeling and Clustering of Textual Data
This project process documents into matrix using TF-IDF. After clustering similar groups of documents, each class is labeled using common key words. The result is visualized and evaluated using precision/recal/F1score and confusion matrix.

## HW4 
- KNN (and fuzzy KNN code) implemented in (HW4 updated) main_notebook.ipynb notebook
- TF-IDF code edited in read.py (from HW3) to process unknown files and recalculated TF-IDF table
- Labels and evaluation of KNN was based on self observation and decision after manually reading the documents

### Run project locally
1. Clone project

2. Create a virtual environment
```
virtualenv .venv
source .venv/bin/activate 
```
3. Install all requirements for the environment
```
pip install -r requirements.txt
```
This makes sure we have compatible package version
4. Run the program
```
python main.py
```

### 
For HW3:
Topics of the folders are summarize in the topics.txt file 

Visualization and evaluation results can be seen directly without running the program in main_notebook.ipynb


