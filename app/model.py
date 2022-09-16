#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 

#importing dataset
dataset=pd.read_csv('hiring.csv')
print(dataset)
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)
X=dataset.iloc[:,:3]

#Converting words to integer values
def conver_to_int(word):
    word_dict={'one':1, 'two':2, 'three':3,'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 
                'ten':10, 'eleven':11,'twelve':12,'zero':0, 0:0}
    return word_dict[word]
X['experience']=X['experience'].apply(lambda x: conver_to_int(x))
y=dataset.iloc[:,-1]

#Splitting training and test set
#since we have a very dataset, we will train our model with all available data

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

#fitting model with training data
regressor.fit(X,y)

#Saving model
pickle.dump(regressor,open("model.pkl","wb"))

#Loading model to compare the results
model=pickle.load(open("model.pkl","rb"))
print(model.predict([[2,9,7]]))
 