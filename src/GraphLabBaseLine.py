
# coding: utf-8

# In[237]:

import graphlab
graphlab.product_key.set_product_key('5C18-F62D-903A-3E36-A80C-BDD9-788A-558B')
import pandas as pd
import sys
get_ipython().magic('pylab inline')


# # Features extraction

# In[200]:

def features_extraction(num=1):
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    items = pd.read_csv('ml-100k/u.item', sep='|', 
                        names=['movie_id', 'movie title' ,'release date','video release date', 
                               'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 
                               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], 
                        encoding='latin-1')
    
    items['release date'] = pd.to_datetime(items['release date'])
    items['release date'] = items['release date'].apply(lambda x: x.year)
    items['release date'] -= items['release date'].min()
    items = items.fillna(int(items['release date'].median()))

    users.sex = users.sex.apply(lambda x: True if x == 'M' else False)
    
    items_data = graphlab.SFrame(items[['movie_id', 'movie title' ,'release date',
       'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']])
    
    users_data = graphlab.SFrame(
        pd.concat([users.drop(['occupation', 'zip_code'], axis=1), 
                   pd.get_dummies(users.occupation)], axis=1)
    )
    
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ml-100k/u' + str(num) + '.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/u' + str(num) + '.test', sep='\t', names=r_cols, encoding='latin-1')

    train_data = graphlab.SFrame(ratings_base)
    test_data = graphlab.SFrame(ratings_test)
    return train_data, test_data, users_data, items_data


# In[230]:

users.occupation.value_counts()


# # Models

# In[267]:

def train_test(train_data, test_data, users_data, items_data):
    models = dict()
#     models['popularity'] = graphlab.popularity_recommender.create(train_data, 
#                                                                   user_id='user_id', 
#                                                                   item_id='movie_id',
#                                                                   target='rating'
#                                                                  )

                                                              
#     models['item_sim'] = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', 
#                                                                  item_id='movie_id', 
#                                                                  target='rating', 
#                                                                  similarity_type='pearson')
                                           
                                                              
    models['fact'] = graphlab.factorization_recommender.create(train_data, 
                                                           user_id='user_id', 
                                                           item_id='movie_id', 
                                                           target='rating', 
                                                           user_data=users_data, 
                                                           item_data=items_data,
                                                           verbose=False,
                                                           regularization=1e-03,
#                                                            num_factors=8
                                                          )     
                              
    predicted = dict()
    for i in models.keys():
        predicted[i] = models[i].predict(test_data)
       
    rmse = dict()
    for i in models.keys():
        rmse[i] = graphlab.evaluation.rmse(predicted[i], test_data['rating'])                                                  
    
    return rmse


# In[268]:

rmse = []
for i in range(1, 6):
    train_data, test_data, users_data, items_data = features_extraction(i)
    rmse.append(train_test(train_data, test_data, users_data, items_data))


# In[269]:

a = [[i[key] for i in rmse]for key in rmse[0].keys()]


# In[270]:

rmse[0].keys()


# In[271]:

asanyarray(a).mean()


# In[256]:

asanyarray(a).mean()


# In[245]:

a


# In[241]:

a

