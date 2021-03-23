import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
datafile= pd.read_csv('Dataset.csv')
datafile.head()
# Renaming DiabetesPedigreeFunction as DPF
datafile = datafile.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
copied = datafile.copy(deep=True)
copied[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = copied[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
copied['Glucose'].fillna(copied['Glucose'].mean())
copied['BloodPressure'].fillna(copied['BloodPressure'].mean())
copied['SkinThickness'].fillna(copied['SkinThickness'].median())
copied['Insulin'].fillna(copied['Insulin'].median())
copied['BMI'].fillna(copied['BMI'].median())

# Model Building

X = datafile.drop(columns='Outcome')
Y = datafile['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
classifier=DecisionTreeClassifier()
classifier = classifier.fit(X_train,Y_train)

features = ['Pregnancies','SkinThickness', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DPF']
accuracies=[]
farray=[]
for i in range(len(features)):
    A = datafile[[features[i]]]
    B = datafile.Outcome
    A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=5)
    classify=DecisionTreeClassifier()
    classify = classify.fit(A_train,B_train)
    prediction = classify.predict(A_test)
    Accuracy=metrics.accuracy_score(B_test,prediction)
    accuracies.append((features[i],Accuracy))
    farray.append(Accuracy)

for i in range(len(features)):
    x = datafile[[features[i]]]
    plt.hist(x, bins=50)
    plt.gca().set(title='Test'+str(i+1)+' Histogram',xlabel=features[i], ylabel='frequency') 
    plt.savefig('static/p'+str(i+1)+'.png')
    plt.clf()
    plt.cla()
    plt.close() 
for i in range(len(features)):
    x = datafile[[features[i]]]
    plt.hist(x, bins=50)
    plt.gca().set(title='All Tests Data Distribution Histogram',xlabel='Values', ylabel='frequency') 
plt.savefig('static/AllinOne.png')


plt.clf()
plt.cla()
plt.close() 
pickle.dump(classifier, open('Dataset.pkl', 'wb'))
pickle.dump(accuracies, open('Accuracies.pkl', 'wb'))

xs = ['Preg', 'SkinThick', 'Insulin', 'BMI', 'Age', 'Glucose', 'BP', 'DPF']
cross_val = pd.DataFrame({'Name': xs, 'Score': farray})
axis = sns.barplot(x = 'Name', y = 'Score', data =cross_val)
axis.set(xlabel='Tests', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
plt.savefig('static/new_plot.png')