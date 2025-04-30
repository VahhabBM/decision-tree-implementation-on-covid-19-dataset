# import the libraries that we need:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#loadig the dataset using pandas:
df = pd.read_csv('Covid Data.csv')
# preproccessing:
df = df.drop(columns=['INTUBED', 'ICU'])  #columns that we do not need
df['PREGNANT'] = df['PREGNANT'].replace([97, 98, 99], 2)  #we change the 'PREGNANT' missing values to 2

# x is the datas without target
X = df.drop(columns=['CLASIFFICATION_FINAL']) 
y = df['CLASIFFICATION_FINAL']  #and then the target

#we divide the datas to two parts:1.the test datas & 2. the train datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#making the tree for our datas
clf = DecisionTreeClassifier(max_depth=18, random_state=42)
clf.fit(X_train, y_train)

#drawing the tree using mathplotlib library
plt.figure(figsize=(20, 10))
class_names = [str(cls) for cls in y.unique()]
plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True)
# plt.show()

y_prediction=clf.predict(X_test)
f1 = f1_score(y_test , y_prediction , average='weighted')
print('f1-score:' , f1)