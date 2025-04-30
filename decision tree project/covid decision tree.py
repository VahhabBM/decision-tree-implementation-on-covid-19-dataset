import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("error",category=RuntimeWarning)

# saving the dataframe to a variable
df=pd.read_csv('Covid Data.csv')
# dropping unwanted columns
df=df.drop(columns='INTUBED')
df=df.drop(columns='ICU')
df['PREGNANT'] = df['PREGNANT'].replace([97, 98, 99], 2) #replacing the missing values with 2
print(df.head())
# spliting the test data(the first 1000 datas)
df_tst=df.iloc[:1000] 
df_test=df_tst.to_dict(orient='records')
y_test=df_tst['CLASIFFICATION_FINAL'].tolist()
# spliting the train data from the test datas 
df=df.iloc[1000:]
# spliting the targets column
targets= df['CLASIFFICATION_FINAL'].tolist()

# the calculation part
# -------------------------------------------------------------------------------------------------------------------------------
# calculates entropy
def entropy(data):
    data = np.array(data)
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy_val = (-1) * np.sum(probabilities * np.log2(probabilities))
    return entropy_val
entropy_kol = entropy(targets)
# calculates information gain
def information_gain(attribute_inpt, data_inpt):
    values_to_find= [98,97,99]
    data1 = np.array(data_inpt)
    attribute1 = np.array(attribute_inpt)
    indexes_to_del=np.where(np.isin(attribute1, values_to_find))[0]
    attribute=np.delete(attribute1 , indexes_to_del)
    data=np.delete(data1 , indexes_to_del)
    entropy_s = entropy(data)
    unique_vals, counts = np.unique(attribute, return_counts=True)
    x = np.sum( np.fromiter( ((counts[i]/len(attribute)) * entropy(data[attribute==unique_vals[i]]) for i in range(len(unique_vals))) , dtype=float ) )
    return entropy_s - x
# calculates split info
def split_info(column):
    column = np.array(column)
    _, counts = np.unique(column, return_counts=True)
    probabilities = counts / len(column)
    splitinfo_val = -np.sum(probabilities * np.log2(probabilities))
    return splitinfo_val
# calculates gain ratio
def gain_ratio(attribute, data):
    split_info_val = split_info(attribute) 
    try:
        gainratio= information_gain(attribute, data) / split_info_val
        return gainratio
    except RuntimeWarning:
        return 0
# -------------------------------------------------------------------------------------------------------------------------------
# ""AGE part(the regression attribute)""
# age is regression so we need to find the treshold for that:
# first we sort datas by age
def age_sorter(dataframe):
    new_dataframe=dataframe[['AGE','CLASIFFICATION_FINAL']]
    return new_dataframe.sort_values(by='AGE')
# we calculate imformation gain and split info for age with another def:
def age_information_gain(age,target , treshold):
    entropy_s=entropy_kol
    target_array=np.array(target)
    age_array=np.array(age)
    left_ages= age_array[age_array <= treshold]
    right_ages= age_array[age_array > treshold]
    treshold_index= len(left_ages)
    left_targets= target_array[: treshold_index]
    list_left_targets=left_targets.tolist()
    right_targes= target_array[treshold_index :]
    list_right_targets=right_targes.tolist()
    x= ( (len(left_ages)/len(target_array)) * (entropy(list_left_targets)) ) + ( (len(right_ages)/len(target_array)) * (entropy(list_right_targets)) )
    return (entropy_s - x)
def age_split_info(attribute , treshold):
    attribute_array=np.array(attribute)
    left_ages=attribute_array[attribute_array <= treshold]
    righ_ages=attribute_array[attribute_array > treshold]
    splt = ( (len(left_ages)/len(attribute)) * math.log2(len(left_ages)/len(attribute)) ) + ( (len(righ_ages)/len(attribute)) * math.log2(len(righ_ages)/len(attribute)) )
    return (splt) * (-1)
#gain ratio calculation for age:
def age_gain_ratio(age , target , treshold):
    a=age_information_gain(age , target , treshold)
    b=age_split_info(age,treshold)
    return a/b
# and finally we define a def which calculates the treshold of our 
def find_treshold(dataframe):
    sorted_dataframe=age_sorter(dataframe)
    age_list=sorted_dataframe['AGE'].tolist()
    target_list=sorted_dataframe['CLASIFFICATION_FINAL'].tolist()
    tresholds_list=[(i-0.5) for i in range(1,122)]
    i_g_for_tresholds=[]
    print("calculting the best treshold for AGE , PLEASE WAIT!")
    for i in tresholds_list:
        i_g_for_tresholds.append(age_information_gain(age_list , target_list , i))
        print(f"information gain for treshold ({i}):" , age_information_gain(age_list , target_list , i))
    treshold_list_index= i_g_for_tresholds.index(max(i_g_for_tresholds))
    treshold_val = tresholds_list[treshold_list_index] 
    return ((treshold_val))
age_treshold = find_treshold(df)
print('please WAIT! runnig this code takes about 180 seconds')
age_target_list = age_sorter(df)['CLASIFFICATION_FINAL'].tolist()
# ----------------------------------------------------------------------------------------------------------------------------------------------
# defining a def which clculates 'gain ratio' for each attribute and compares them and finally , it gives you the root!
def find_root(dataframe):
    list_of_categories = {column: dataframe[column].tolist() for column in dataframe.columns}
    target_list = list_of_categories['CLASIFFICATION_FINAL']
    list_of_categories.pop('CLASIFFICATION_FINAL', None)
    g_r = {}
    for key in list_of_categories:
        if key != 'AGE':
            g_r[key] = gain_ratio(list_of_categories[key], target_list)
        elif key == 'AGE': 
            g_r['AGE'] = age_gain_ratio(list_of_categories['AGE'], age_target_list, age_treshold)
    if not g_r:
        return None
    root_g_r = max(g_r.values())
    for k, v in g_r.items():
        if v == root_g_r:
            list_of_categories.pop(k)
            return k
    return None 
# ----------------------------------------------------------------------------------------------------------------------------------------------
# now , it's time to define the decision tree
# first we define a Node class:
class Node:
    def __init__(self, root=None, classification=None):
        self.root = root
        self.classification = classification
        self.child = {}
        self.is_leaf = classification is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, dataset):
        self.tree = self.build_tree(dataset)
        self.default_class = max(set(dataset['CLASIFFICATION_FINAL']), key=dataset['CLASIFFICATION_FINAL'].tolist().count)

    def build_tree(self, dataset, depth=0):
        target_column = 'CLASIFFICATION_FINAL'
        targets_list = dataset[target_column].tolist()
        if depth >= self.max_depth or len(set(targets_list)) <= 1:
            classification = max(set(targets_list), key=targets_list.count) if len(targets_list) != 0 else self.default_class
            return Node(classification=classification)
        root = find_root(dataset)
        if root is None:
            if len(targets_list) !=0:
                classification =set(targets_list)
                return Node(classification=classification)
            else:
                return Node(classification=None)
        node = Node(root=root)
        if root == 'AGE':
            threshold = age_treshold
            left_data = dataset[dataset[root] <= threshold].drop(columns=[root])
            right_data = dataset[dataset[root] > threshold].drop(columns=[root])
            node.child['left'] = self.build_tree(left_data, depth + 1)
            node.child['right'] = self.build_tree(right_data, depth + 1)
        else:
            for value in dataset[root].unique():
                child_data = dataset[dataset[root] == value].drop(columns=[root])
                node.child[value] = self.build_tree(child_data, depth + 1)
        return node
# and at the end of our tree class, we define a predictor def which uses the tree object to predict the inputed values
    def predict(self, sample):
        node = self.tree
        while not node.is_leaf:
            if node.root == 'AGE':
                threshold = age_treshold
                if sample['AGE'] <= threshold:
                    node = node.child['left']
                else:
                    node = node.child['right']
            else:
                value = sample[node.root]
                if value in node.child:
                    node = node.child[value]
                else:
                    return self.default_class
        return node.classification

# making instances of the classes
decision_tree= DecisionTree(max_depth=16)
decision_tree.fit(df)
print('the decision tree has been maden')
# and then , the predict part!
y_pred=[]
sample_data = df_test
for i in sample_data:
    prediction = decision_tree.predict(i)
    y_pred.append(prediction)
f1=f1_score(y_test , y_pred , average='weighted')
print( 'f1_score:',f1)

