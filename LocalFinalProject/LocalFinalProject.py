from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing as pp

from pybbn.generator.bbngenerator import convert_for_drawing
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from imblearn.over_sampling import RandomOverSampler
from shutil import get_terminal_size
from csv import reader

import sklearn.utils._weight_vector
import sklearn.neighbors._typedefs
import sklearn.neighbors._quad_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd
import networkx as nx
import warnings


pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 1000)
csv = pd.read_csv("Kickstarter001.csv")

#print(csv)
#preprocessing

#reduce columns
keep_col = ['backers_count', 'goal', 'pledged', 'state', 'created_at', 'deadline', 'category',  'name', 'staff_pick']
data = csv[keep_col]
category_types = []
le = pp.LabelEncoder();
lb = pp.LabelBinarizer()
os = RandomOverSampler(sampling_strategy='minority')

#new_csv.head()

#encode data
# ------------------------------------------------------------------------
# DATA PREPROCESSING
def encode_data(data):

    data = pd.DataFrame(data)
    

    #(in backers)
    #backers_count_encoded = pd.cut(data.backers_count, bins=[-1, 10, 100,1000,10000, 100000, 1000000], labels=[0,1,2,3,4,5])
    backers_count_encoded = pd.cut(data.backers_count, bins=[-1, 100, 1000, 1000000], labels=[0,1,2])
    data.insert(0, 'backers_count_encoded', backers_count_encoded)

    #(in dollars)
    #goal_encoded = pd.cut(data.goal, bins=[-1,10, 100, 1000, 10000, 100000, 1000000], labels=[0,1,2,3,4,5])
    goal_encoded = pd.cut(data.goal, bins=[-1, 1000, 50000, 1000000], labels=[0,1,2])

    data.insert(1, 'goal_encoded', goal_encoded)

    #(in dollars)
    pledged = pd.cut(data.pledged, bins=[-1, 1000, 50000, 1000000], labels=[0,1,2])
    data.insert(2, 'pledged_encoded', pledged)

    #(succeeded, failed, canceled)
    state_encoded = list(le.fit_transform(data.state))
    data.insert(3, 'state_encoded', state_encoded)

    #calculate duration from deadline and created_at
    period = []
    zipped_period = zip(list(data.deadline), list(data.created_at))
    for dl, ca in zipped_period:
        period.append(dl - ca)
    #(in seconds) 25 hours, 33 days, 6 months (compensate for wiggle room)
    #period_encoded = pd.cut(period, bins=[-1, 90000, 691200, 2851200, 15552000], labels=[0,1,2,3])
    period_encoded = pd.cut(period, bins=[-1, 90000, 2851200, 15552000], labels=[0,1,2])
    data.insert(4, 'period', period_encoded)


    #(in parent_category) parse category info (json string) for parent category informaion, and then encode that into final table
    category_raw = list(data.category)
    category = []
    for c in category_raw:
        i_adjust = 15;
        index = c.find('"parent_name"')
        if (index == -1):
            i_adjust = 8
            index = c.find('"name"')
        index_end = c.find(',', index)
        parent_category = c[index+i_adjust:index_end-1]
        category.append(parent_category)
        if parent_category not in category_types:
            category_types.append(parent_category)
    
    category_encoded = le.fit_transform(category)
    #print(category_encoded)
    data.insert(5, 'category_encoded', category_encoded)

    #delete legacy columns (pre-processed columns)
    del data['backers_count'], data['goal'], data['state'], data['pledged'], data['created_at'], data['deadline'], data['category']

    data.dropna(inplace= True)

    return data

#confusion matrix values
#to avoid the 0 probability error, add one to each value upon return
def get_cm_values(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


# BAYESIAN BELIEF NETWORK CODE
def bayesian_belief_network():

    deadline = BbnNode(Variable(0, 'deadline', ['long', 'medium','short']), [0.33, 0.33, 0.33])
    goal = BbnNode(Variable(1, 'goal', ['large', 'medium', 'small']), [0.33, 0.33, 0.33])
    backers = BbnNode(Variable(3, 'backers', ['many', 'medium', 'few']), [0.6,0.3,0.2,0.3,0.4,0.4,0.1,0.3,0.4])
    pledges = BbnNode(Variable(4, 'pledges', ['large', 'medium', 'small']), [0.6,0.3,0.1,0.2,0.5,0.2,0.2,0.2,0.7])  
    state = BbnNode(Variable(5, 'state', ['success', 'fail']), [0.7,0.5,0.2,0.8,0.6,0.2,0.9,0.7,0.3,0.3,0.5,0.8,0.2,0.4,0.8,0.1,0.3,0.7])

    
    global Bbn
    Bbn = Bbn() \
    .add_node(deadline) \
    .add_node(goal) \
    .add_node(backers) \
    .add_node(pledges) \
    .add_node(state) \
    .add_edge(Edge( goal, backers, EdgeType.DIRECTED)) \
    .add_edge(Edge( goal, pledges, EdgeType.DIRECTED)) \
    .add_edge(Edge( backers, state, EdgeType.DIRECTED)) \
    .add_edge(Edge( deadline, state, EdgeType.DIRECTED))
   

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        graph = convert_for_drawing(Bbn)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    
        #uncomment this section to print BBN
        #plt.figure(figsize=(8, 8))
        #plt.subplot(121)
        #labels = dict([(k, node.variable.name) for k, node in Bbn.nodes.items()])
        #nx.draw(graph, pos=pos, with_labels=True, labels=labels)
        #plt.title('Bayesian Belief Network Diagram')
        #plt.show()

    join_tree = InferenceController.apply(Bbn)

    print("BBN - Original Probabilities")
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print(node)
        print(potential)
        print('--------------------->')

    # insert an observation evidence
    ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name('goal')) \
        .with_evidence('large', 1.0) \
        .build()
    join_tree.set_observation(ev)
    #print the marginal probabilities
    
    print()
    print()
    print("BBN - Set Goal = Large")
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print(node)
        print(potential)
        print('--------------------->')
    

#-------------------------------------------------------------------------------------------
# Run Process

bayesian_belief_network()

#encode data and set class labels
data = pd.DataFrame(encode_data(data))
#object names list
object_names = list(le.fit_transform(data.name))
#class label list
class_labels = list(le.fit_transform(data.staff_pick))


category_data = []
category_class_labels = []
for i in range(len(category_types)):
    #append a table containing all of current category data
    category_data.append(data.loc[data['category_encoded'] == i])
    category_class_labels.append(list(le.fit_transform(category_data[i].staff_pick)))
    del category_data[i]['name'], category_data[i]['staff_pick']
    #print(len(category_data[i]))


del data['name'], data['staff_pick']

n_splits = 10
gnb = GaussianNB()
svml = svm.SVC(kernel='linear', class_weight='balanced')

#------------------------------------------------------------------------
# PART 1
print()
print()
print()
print("Scenerio One: 10-Fold Cross Validation on Complete Dataset")
# Use Cross Validation - 10 Fold Split
kf = KFold(n_splits=n_splits)
print(kf)
kfold_iteration = 0
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train = []
    y_test = []
    for i in train_index:
        y_train.append(class_labels[i])
    for i in test_index:
        y_test.append(class_labels[i])  
    

    y_pred_GNB = gnb.fit(X_train, y_train).predict(X_test)
    #TP, FP, FN, TN = get_cm_values(y_test, y_pred_GNB)
    #GNB_accuracy = (TP+TN)/(TP+FP+FN+TN)
    #GNB_precision = TP/(TP+FP)
    #GNB_recall = TP/(TP+FN)

    a = metrics.f1_score(y_test, y_pred_GNB)
    b = metrics.precision_score(y_test, y_pred_GNB)
    c = metrics.recall_score(y_test, y_pred_GNB)
    
    #print(metrics.confusion_matrix(y_test, y_pred_GNB)) 
    print("GNB Accuracy:",a)
    print("GNB Precision:",b)
    print("GNB Recall:",c)
    
    #scale and overfit data for LSVC
    X_scaled, X_test2 = pp.scale(X_train), pp.scale(X_test)
    X_over_train, y_over_train = os.fit_resample(X_scaled, y_train)
    X_over_test, y_over_test = os.fit_resample(X_scaled, y_train)
    
    #print(X_over_train[0:10])

    y_pred_svml = svml.fit(X_over_train, y_over_train).predict(X_over_test)
    #TP2, FP2, FN2, TN2 = get_cm_values(y_over_test, y_pred_svml)
    #svml_accuracy = (TP2+TN2)/(TP2+FP2+FN2+TN2)
    #svml_precision = TP2/(TP2+FP2)
    #svml_recall = TP2/(TP2+FN2)

    d = metrics.f1_score(y_over_test, y_pred_svml)
    e = metrics.precision_score(y_over_test, y_pred_svml)
    f = metrics.recall_score(y_over_test, y_pred_svml)
    
    #print(metrics.confusion_matrix(y_over_test, y_pred_svml))
    print("SVMl Accuracy:",d)
    print("SVMl Precision:",e)
    print("SVMl Recall:",f)

    print("-----------------------------------------------------")
    kfold_iteration = kfold_iteration+1


#----------------------------------------------------------------------
# PART 2
print()
print()
print()
print("Scenerio Two: 10-Fold Cross Validation on Complete Dataset")
#Top 3 highest appearing categories:
#6, 10, 1, 5, 13 (5, 9, 0, 4, 12)
#Journalism, Publishing, Film & Video, (Fashion, Games)

#print(category_types)
topthree = [5, 9, 0] # 4, 12] #not enough data for Fashion and Games
aprTable = [[]]

for i in topthree:
    X_train, X_test, y_train, y_test = train_test_split(category_data[i], category_class_labels[i], test_size=0.1, random_state=0)
    results_matrix = [[]]

    y_pred_GNB = gnb.fit(X_train, y_train).predict(X_test)
    #TP, FP, FN, TN = get_cm_values(y_test, y_pred_GNB)
    #GNB_accuracy = (TP+TN)/(TP+FP+FN+TN)
    #GNB_precision = TP/(TP+FP)
    #GNB_recall = TP/(TP+FN)
    
    a = metrics.accuracy_score(y_test, y_pred_GNB)
    b = metrics.precision_score(y_test, y_pred_GNB)
    c = metrics.recall_score(y_test, y_pred_GNB)
    y = metrics.f1_score(y_test, y_pred_GNB)

    print("for category:",category_types[i])
    print("category GNB Accuracy:",a)
    print("category GNB Precision:",b)
    print("category GNB Recall:",c)
    print("category GNB F1:",y)
    print(metrics.confusion_matrix(y_test, y_pred_GNB))
    
    #scale the data and oversample for LSVC
    X_scaled, X_test2 = pp.scale(X_train), pp.scale(X_test)
    X_over_train, y_over_train = os.fit_resample(X_scaled, y_train)
    X_over_test, y_over_test = os.fit_resample(X_scaled, y_train)

    y_pred_svml = svml.fit(X_over_train, y_over_train).predict(X_over_test)
    #TP2, FP2, FN2, TN2 = get_cm_values(y_over_test, y_pred_svml)
    #svml_accuracy = (TP2+TN2)/(TP2+FP2+FN2+TN2)
    #svml_precision = TP2/(TP2+FP2)
    #svml_recall = TP2/(TP2+FN2)

    d = metrics.accuracy_score(y_over_test, y_pred_svml)
    e = metrics.precision_score(y_over_test, y_pred_svml)
    f = metrics.recall_score(y_over_test, y_pred_svml)
    z = metrics.f1_score(y_over_test, y_pred_svml)
    print("category SVMl Accuracy:",d)
    print("category SVMl Precision:",e)
    print("category SVMl Recall:",f)
    print("category SVMl F1:",z)

    print(metrics.confusion_matrix(y_over_test, y_pred_svml))

    print("------------------")
    print();

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    X = np.arange(3)
    aprTable.append([a,b,c,d,e,f])





