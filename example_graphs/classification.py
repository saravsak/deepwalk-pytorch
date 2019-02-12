# coding: utf-8

# In[ ]:


#!/usr/bin/env python

'''
READ ME:
The following message to understand the NEW classification script (written in python3).
How to use this script:
There are two ways of using this script:
1) As an import
OR
2) As a stand alone script
----------------------------------------------
Using this as an imported module:
1) Add "import classification"
2) Use the classify function by calling it:
   classification.classify(...) ----> Note ... are the arguments you need to set
3) The classify function takes the following arguments:
    Required arguments:
        - "emb": The path and name of the embeddings file, type:string
        - "network": The path and name of the .mat file containing the adjacency matrix and node labels of the input network, type:string
        - "dataset": The name of your dataset (used for output), type:string
        - "algorithm", The name of the algorithm used to generate the embeddings (used for output), type:string
    Optional arguments
        - "num_shuffles": The number of shuffles for training, type:int
            - default value: 5
        - "writetofile": If true, output classification results to file'), type:bool
            - default value: True
        - "adj_matrix_name": The name of the adjacency matrix inside the .mat file, type:string
            - default value: "network"
        - "word2vec_format": If true, genisim is used to load the embeddings, type:bool
            - default value: True
        - "embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict
            - default value: {}
        - "training_percents": List of split "percents" for training and test sets '), type:list
            - default value: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        - "label_matrix_name": The name of the labels matrix inside the .mat file, type:string
            - default value: "group"
        - "classifier": Classifier to be used; Choose from "LR" or "SVM", type:string
            - default value = "LR"
        - "test_kernel": The kernel to use for the SVM/SVC classifer (not improtant for LR classifier); Choose from "linear" or "rbf", type:string
            - default value = "LR"
        - "grid_search": If true, best parameters are found for classifier (increases run time), type:bool
            - default value = True
        - "output_dir": Specify the path to store results, type:str
            - default value = "./"
Example for LR classification:
#classification.classify(emb="../emb/blogcatalog.emb", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", word2vec_format=True, embedding_params={"walk" :10, "p" : 1})
Example for SVM (linear kernel) classification:
#classification.classify(emb="../emb/embed_blogcatalog.npy", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", classifier="SVC", word2vec_format=False, embedding_params={"walk" :10, "p" : 1})
Example for SVM (rbf kernel) classification:
#classification.classify(emb="../emb/embed_blogcatalog.npy", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", classifier="SVC", test_kernel="rbf", word2vec_format=False, embedding_params={"walk" :10, "p" : 1})
----------------------------------------------
Using this as a stand alone module/script:
1) Can be called from command line with the following arugments:
	# Required arguments (must be given in the correct order):
	"--emb", type=string, required=True, help='The path and name of the embeddings file
	"--network", type=string, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network
	"--dataset", type=string,required=True, help='The name of your dataset (used for output)
	"--algorithm", type=string, required=True, help='The name of the algorithm used to generate the embeddings (used for output)
	# Flags (use if true, they don't require additional parameters):
	"--writetofile": If used, classification results to are written to a file
	"--word2vec_format": If used, genisim is used to load the embeddings
	# Optional arguments
	"--num_shuffles", default=5, type=int, help='The number of shuffles for training'
	"--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file'
	"--word2vec_format", action="store_false", help='If true, genisim is used to load the embeddings'
	"--embedding_params", type=json.loads, help='"embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict
	"--training_percents", default=training_percents_default, type=arg_as_list,  help='List of split "percents" for training and test sets (i.e. [0.1, 0.5, 0.9]')
	"--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file'
	"--classifier", default="LR",choices=["LR","SVM"], help='Classifier to be used; Choose from "LR" or "SVM"'
	"--test_kernel", default="linear",choices=["linear","rbf"], help='Kernel to be used for SVM classifier; Choose from "linear" or "rbf"'
	"--output_dir", default="./", type=str, help='Specify the path to store results'
Example for LR classification:
python classification.py --emb ../emb/blogcatalog.npy --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --writetofile
Example for SVM (linear kernel) classification:
python classification.py --classifier SVM --test_kernel linear --emb ../emb/blogcatalog.emb --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --word2vec_format --grid_search
Example for SVM (rbf kernel) classification:
python classification.py --classifier SVM --test_kernel rbf --emb ../emb/blogcatalog.emb --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --word2vec_format --grid_search
----------------------------------------------
----------------------------------------------
Improtant things to keep in mind:
1) Embedding format must either be in word2vec format
    OR
   Each row of a file must be an embedding of a node in the graph (in the correct order of the mat file)
2) If you get memory errors with grid search, set the agrument to false or don't use the flag (depends on how you use the flag)
3) Keep grid search on
'''


# In[1]:


import ast
import sys
import json
import numpy
import pandas
import shutil
import os.path
import sklearn
import logging
import warnings
import scipy as sp
import scipy.sparse
import multiprocessing
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

from six import iteritems
from scipy.io import loadmat
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle as skshuffle
from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


# Set seed:
numpy.random.seed(42)



program = os.path.basename(sys.argv[0])

logger = logging.getLogger(program)
logging.root.setLevel(level=logging.INFO)
logger.info("Running %s", ' '.join(sys.argv))

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


# In[4]:


training_percents_default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# In[5]:


# Checks if a string is a valid list
def arg_as_list(string_list):
  parsed_list = ast.literal_eval(string_list)
  if type(parsed_list) is not list:
    raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (string_list))
  return parsed_list


# In[6]:


def load_graph(matfile, adj_matrix_name, label_matrix_name):
  mat = loadmat(matfile)
  labels_matrix = mat[label_matrix_name]
  labels_sum = labels_matrix.sum(axis=1)
  indices = numpy.where(labels_sum>0)[0]
  labels_matrix = sp.sparse.csc_matrix(labels_matrix[indices])
  A = mat[adj_matrix_name][indices][:,indices]
  graph = sparse2graph(A)
  labels_count = labels_matrix.shape[1]
  multi_label_binarizer = MultiLabelBinarizer(range(labels_count))
  return mat, A, graph, labels_matrix, labels_count, multi_label_binarizer, indices


# In[7]:


# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
def load_embeddings(embeddings_file, word2vec_format, graph, indices):
  model = None
  features_matrix = None

  if word2vec_format:
    if ".bin" in embeddings_file:
      model = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
    else:
      model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = numpy.asarray([model[str(node)] for node in indices])

  else:
    features_matrix = numpy.load(embeddings_file)[indices]

  normalized_features_matrix = normalize(features_matrix, norm="l2")

  return features_matrix, normalized_features_matrix


# In[8]:


def sparse_tocoo(temp_y_labels):
  y_labels = [[] for x in range(temp_y_labels.shape[0])]
  cy =  temp_y_labels.tocoo()
  for i, j in zip(cy.row, cy.col):
    y_labels[i].append(j)
  assert sum(len(l) for l in y_labels) == temp_y_labels.nnz
  return y_labels


# In[9]:


def sparse2graph(x):
  G = defaultdict(lambda: set())
  cx = x.tocoo()
  for i,j,v in zip(cx.row, cx.col, cx.data):
    G[i].add(j)
  return {str(k): [str(x) for x in v] for k,v in iteritems(G)}


# In[10]:


def calc_metrics(normalized, dataset, algorithm, num_shuffles, all_results, writetofile, embedding_params, emb_size, clf, output_dir_and_classifier_name):

    columns=["Algorithm", "Dataset", "Train %", "Normalized Embeddings", "Micro-F1", "Macro-F1", "Accuracy", "Num of Shuffles", "Embedding Size", "Classifier"]

    if embedding_params != None:
        columns = columns + list(embedding_params.keys())
    print ('-------------------')
    if writetofile:
        results_df = pandas.DataFrame(columns=columns)

    print (",".join(columns))
    for train_percent in sorted(all_results.keys()):

        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:

            for metric, score in iteritems(score_dict):
                avg_score[metric] += score

        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])

        if writetofile:
            temp = {
                     "Dataset": dataset,
                     "Train %": train_percent,
                     "Normalized Embeddings": str(normalized),
                     "Algorithm": algorithm,
                     "Micro-F1": avg_score["micro"],
                     "Macro-F1": avg_score["macro"],
                     "Accuracy": avg_score["accuracy"],
                     "Num of Shuffles": num_shuffles,
                     "Embedding Size":    emb_size,
                     "Estimator": str(clf.get_params()["estimator"])
                    }
            if embedding_params != None:
                temp.update(embedding_params)
            results_df = results_df.append(temp, ignore_index=True)

        clf_params_string = str(clf.get_params()["estimator"]).replace("\n", "")

        embedd_params_file_name = ""
        if embedding_params != None:
            embedding_params_string = ""
            for key in embedding_params.keys():
                embedd_params_file_name = embedd_params_file_name + "_" + key + "_" + str(embedding_params[key])
                if embedding_params_string == "":
                    embedding_params_string = str(embedding_params[key])
                else:
                    embedding_params_string = embedding_params_string + "," + str(embedding_params[key])
            print ("{},{},{},{},{},{},{},{},{},{},{}".format(algorithm, dataset, normalized, train_percent,                 avg_score["micro"], avg_score["macro"], avg_score["accuracy"],                 num_shuffles, emb_size, clf_params_string, embedding_params_string))
        else:
            print ("{},{},{},{},{},{},{},{},{},{}".format(algorithm, dataset, normalized, train_percent,                 avg_score["micro"], avg_score["macro"], avg_score["accuracy"], num_shuffles,                 emb_size, clf_params_string))

    if writetofile:
        output_file_obj = open(output_dir_and_classifier_name + "_" + dataset + "_" + "classi_results" + "_" + algorithm + embedd_params_file_name + ".csv", "a")
        results_df.to_csv(output_file_obj, index = False, sep=',', header=output_file_obj.tell()==0)
        print ("File saved at: {}".format(output_dir_and_classifier_name + "_" + dataset + "_" + "classi_results" + "_" + algorithm + embedd_params_file_name + ".csv"))


# In[11]:


def predict_top_k(classifier, X, top_k_list):
    assert X.shape[0] == len(top_k_list)
    probs = numpy.asarray(classifier.predict_proba(X))
    all_labels = []
    for i, k in enumerate(top_k_list):
        probs_ = probs[i, :]
        labels = classifier.classes_[probs_.argsort()[-k:]].tolist()
        all_labels.append(labels)
    return all_labels


# In[12]:


def get_dataset_for_classification(X, y, train_percent):
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=1-train_percent)
    y_train = sparse_tocoo(y_train_)
    y_test = sparse_tocoo(y_test_)
    return X_train, X_test, y_train_, y_train, y_test_, y_test


# In[13]:


def get_classifer_performace(classifer, X_test, y_test, multi_label_binarizer):
    top_k_list_test = [len(l) for l in y_test]
    y_test_pred = predict_top_k(classifer, X_test, top_k_list_test)

    y_test_transformed = multi_label_binarizer.transform(y_test)
    y_test_pred_transformed = multi_label_binarizer.transform(y_test_pred)

    results = {}
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_test_transformed, y_test_pred_transformed, average=average)
    results["accuracy"] = accuracy_score(y_test_transformed, y_test_pred_transformed)

    print ("======================================================")
    print("Best Scores with best params: {}".format(str(classifer.get_params()["estimator"]).replace("\n", "")))
    for metric_score in results:
        print (metric_score, ": ", results[metric_score])
    print ("======================================================")
    return results


# In[14]:


def logistic_regression_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer):
    lf_classifer = OneVsRestClassifier(LogisticRegression(solver='lbfgs'), n_jobs=-1)
    #if grid_search:
    #    parameters = {
    #        "estimator__penalty" : ["l1", "l2"],
    #        "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]
    #    }

    #    lf_classifer = GridSearchCV(lf_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)

    lf_classifer.fit(X_train, y_train_.toarray())
    results = get_classifer_performace(lf_classifer, X_test, y_test, multi_label_binarizer)
    return lf_classifer, results


# In[15]:


def svc_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer, test_kernel):

    if test_kernel == "linear":
        svc_classifer = OneVsRestClassifier(SVC(kernel="linear", probability=True),n_jobs=-1)

    if test_kernel == "rbf":
        svc_classifer = OneVsRestClassifier(SVC(kernel="rbf", probability=True), n_jobs=-1)

    if grid_search:

        if test_kernel == "linear":
            parameters = {
                "estimator__C": [0.01, 0.1, 1, 10, 100, 1000],
            }
        if test_kernel == "rbf":
            parameters = {
                "estimator__C": [0.01, 0.1, 1, 10, 100, 1000],
                "estimator__gamma": [0.001, 0.01, 0.1, 1, 10, 100]
            }

        svc_classifer = GridSearchCV(svc_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)

    svc_classifer.fit(X_train, y_train_.toarray())
    results =  get_classifer_performace(svc_classifer, X_test, y_test, multi_label_binarizer)
    return svc_classifer, results


# In[38]:


def classify(emb="", network="", training_percents=training_percents_default, dataset="", writetofile=True, algorithm="", adj_matrix_name="network", label_matrix_name="group", num_shuffles=5, word2vec_format=True, embedding_params=None, classifier="LR",  test_kernel="linear", grid_search=True, output_dir="./"):

    if emb == "":
        print ("Missing embedding file path.")
        sys.exit(1)
    if network == "":
        print ("Missing network file path.")
        sys.exit(1)
    if type(training_percents) != type([]):
        print ("Training percents must be given in a list format. (i.e. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])")
        sys.exit(1)
    if dataset == "" and writetofile == True:
        print ("Missing dataset name. Dataset name need for file output")
        sys.exit(1)
    if algorithm == "" and writetofile == True:
        print ("Missing algorithm name. Algorithm name need for file output")
        sys.exit(1)

    # 1. Load labels
    mat, A, graph, labels_matrix, labels_count, multi_label_binarizer, indices = load_graph(network, adj_matrix_name, label_matrix_name)

    # 2. Load embeddings
    features_matrix, normalized_features_matrix = load_embeddings(emb, word2vec_format, graph, indices)
    feature_matrices = [(False, features_matrix), (True, normalized_features_matrix)]

    # 3. Fit multi-label binarizer
    y_fitted = multi_label_binarizer.fit(labels_matrix)

    # 5. Store classifier name for file
    classifer_name_for_file = classifier

    if classifier == 'SVM':
        classifer_name_for_file = classifer_name_for_file + "_" + test_kernel

    # 5. Train
    for features_matrix_tuple in feature_matrices:

        # Dict of lists to score each train/test group
        all_results = defaultdict(list)

        normalized = features_matrix_tuple[0]
        features_matrix = features_matrix_tuple[1]

        print ("========================= Normalized Embeddings: " + str(normalized) + " =========================")

        for train_percent in training_percents:

            logger.info("Starting with split %s", train_percent)

            for x in range(num_shuffles):

                logger.info("Shuffle number %s", x)

                # Shuffle, to create train/test groups
                shuf = skshuffle(features_matrix, labels_matrix)

                X, y = shuf
                X_train, X_test, y_train_, y_train, y_test_, y_test = get_dataset_for_classification(X, y, train_percent)


                clf = None
                results ={}
                if classifier == 'LR':
                    clf, results = logistic_regression_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer)

                elif classifier == 'SVM':
                     clf, results = svc_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer, test_kernel)

                all_results[train_percent].append(results)
            logger.info("Done with %s", train_percent)

        calc_metrics(normalized, dataset, algorithm, num_shuffles, all_results, writetofile, embedding_params, X.shape[1], clf, output_dir + classifer_name_for_file)
        print(len(all_results))



def main():
    parser = ArgumentParser("scoring",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')

    # Required arguments
    parser.add_argument("--emb", type=str, required=True, help='The path and name of the embeddings file')
    parser.add_argument("--network", type=str, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network')
    parser.add_argument("--dataset", type=str, required=True, help='The name of your dataset (used for output)')
    parser.add_argument("--algorithm",type=str, required=True, help='The name of the algorithm used to generate the embeddings (used for output)')

    # Optional arguments
    parser.add_argument("--num_shuffles", default=5, type=int, help='The number of shuffles for training')
    parser.add_argument("--writetofile", action="store_true", help='If the flag is set, then the results are written to a file')
    parser.add_argument("--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file')
    parser.add_argument("--word2vec_format", action="store_true", help='If the flag is set, then genisim is used to load the embeddings')
    parser.add_argument("--embedding_params", type=json.loads, help='"embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict')
    parser.add_argument("--training_percents", default=training_percents_default, type=arg_as_list,  help='List of split "percents" for training and test sets (default is [0.1, 0.5, 0.9]')
    parser.add_argument("--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file')
    parser.add_argument("--classifier", default="LR",choices=["LR","SVM"], help='Classifier to be used; Choose from "LR" or "SVM"')
    parser.add_argument("--test_kernel", default="linear",choices=["linear", "rbf"], help='Kernel to be used for SVM classifier; Choose from "linear" or "rbf"')
    # Disabling the grid_search option from command line ... DO NOT UNCOMMENT:
    parser.add_argument("--grid_search", action="store_false", help='If the flag is set, then grid search is NOT used.')
    parser.add_argument("--output_dir", default="./", type=str, help='Specify the path to store results')
    args = parser.parse_args()

    print (args)
    if args.embedding_params == None:
        args.embedding_params = {}
    assert args.embedding_params != None

    classify(emb=args.emb, network=args.network, dataset=args.dataset, algorithm=args.algorithm,             num_shuffles=args.num_shuffles, writetofile=args.writetofile,             classifier=args.classifier, adj_matrix_name=args.adj_matrix_name,             word2vec_format=args.word2vec_format, embedding_params=args.embedding_params,             training_percents=args.training_percents,             test_kernel=args.test_kernel, label_matrix_name=args.label_matrix_name, grid_search=True)



if __name__ == "__main__":
    main()
