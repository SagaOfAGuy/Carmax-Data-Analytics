from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import missingno as msno


# CSV Filepath
filepath="./ShowcaseDataFall21.csv"


# read CSV
dataset = read_csv(filepath)
print(dataset.dtypes)
# Get numerical data
#numerical_data = dataset.select_dtypes(exclude=['int64'])

# Get categorical data
#categorical_data = dataset.select_dtypes(include=['object'])

#print(numerical_data.dtypes)

fig_1 = msno.heatmap(dataset)
fig_1_copy = fig_1.get_figure()
fig_1_copy.savefig('heatmap_nums2.png',bbox_inches='tight')

#print(dataset.dtypes)




#numeric_values = 
#categorical_values = 
#print(dataset.shape)

#fig = msno.matrix(dataset)
#fig_copy = fig.get_figure()
#fig_copy.savefig('matrix.png',bbox_inches= 'tight')


#fig2 = msno.heatmap(filepath)
#fig2_copy = fig2.get_figure()
#fig2_copy.savefig('heatmap.png',bbox_inches='tight')







