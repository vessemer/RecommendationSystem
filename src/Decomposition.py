
# coding: utf-8

# In[23]:

from subprocess import call
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[12]:

options = {
    'SigmoidUserAsymmetricFactorModel': 'num_factors=5 '
                                        'regularization=0.003 '
                                        'bias_reg=0.01 '
                                        'learn_rate=0.006 '
                                        'bias_learn_rate=0.7 num_iter=70',

    'SVDPlusPlus': 'num_factors=50 '
                   'regularization=1 '
                   'bias_reg=0.005 '
                   'learn_rate=0.01 '
                   'bias_learn_rate=0.07 '
                   'num_iter=50 '
                   'frequency_regularization=true',
    
    'SigmoidItemAsymmetricFactorModel': 'num_factors=10 '
                                        'regularization=0.005 '
                                        'bias_reg=0.1 '
                                        'learn_rate=0.006 '
                                        'bias_learn_rate=0.7 num_iter=90',
    
    'BiasedMatrixFactorization': 'num_factors=40 '
                                 'bias_reg=0.1 '
                                 'reg_u=1.0 '
                                 'reg_i=1.2 '
                                 'learn_rate=0.07 '
                                 'num_iter=100 '
                                 'frequency_regularization=true bold_driver=true',
    
#     'ItemKNNPearson': 'k=40 shrinkage=2500 reg_u=12 reg_i=1',
    
#     'UserKNNCosine': 'k=40 reg_u=12 reg_i=1',
#     'UserKNNPearson': 'k=60 shrinkage=25 reg_u=12 reg_i=1',
#     'ItemKNNCosine': 'k=40 reg_u=12 reg_i=1',
    
#     'SVDPlusPlus': 'num_factors=20 '
#                    'regularization=0.1 '
#                    'bias_reg=0.005 '
#                    'learn_rate=0.01 '
#                    'bias_learn_rate=0.007 '
#                    'num_iter=50'
}


# In[13]:

predictions = {}
for i in range(1, 6):
    predictions[str(i)] = []


# In[4]:

for method in options.keys():
    for i in range(1, 6):
        call(['rating_prediction',
              '--training-file=./ml-100k/u' + str(i) + '.base',
              '--test-file=./ml-100k/u' + str(i) + '.test',
              '--recommender=' + method,
              '--recommender-options=' + options[method],
              '--random-seed=42',
              '--no-id-mapping',
              '--prediction-file=./ml-100k/predictions/' + str(i) + '_' + method
             ])


# In[14]:

for method in options.keys():
    for i in range(1, 6):
        predictions[str(i)].append(
            pd.read_csv('./ml-100k/predictions/' + str(i) + '_' + method, 
                        sep='\t', 
                        names=['user_id', 'item_id', 'rating'])
        )
        


# In[34]:

test = []
train = []
cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
for i in range(1, 6):
    train.append(pd.read_csv('ml-100k/u' + str(i) + '.base', sep='\t', names=cols, encoding='latin-1'))
    test.append(pd.read_csv('ml-100k/u' + str(i) + '.test', sep='\t', names=cols, encoding='latin-1'))


# In[37]:

for i in range(1, 6):
    for method, el in enumerate(options.keys()):
        print('RMSE of' + el + ' on ' + str(i) + ' part:', 
              mean_squared_error(test[i - 1].rating, predictions[str(i)][method].rating) ** 0.5)


# In[45]:

rmse = []
for i in range(1, 6):
    grouped = sum([predictions[str(i)][method].rating for method in range(0, 4)]) / 4.
    rmse.append(
        mean_squared_error(test[i - 1].rating, 
                           grouped) ** 0.5)
    print('RMSE of' + el + ' on ' + str(i) + ' part:', rmse[-1])


# In[46]:

sum(rmse) / 5.

