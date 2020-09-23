#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# install turicreate and graphlab create (Machine learning)


# In[ ]:


import graphlab


# In[ ]:


## load some house sales data


# In[ ]:


sales = graphlab.Sframe('home_data.gl/')
sales


# In[ ]:


# use graphlab canvas to view some visualisation


# In[ ]:


graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x ="sqft_living", y='price') # y axixs is price(whatim trying to predict)


# In[ ]:


# create a simple regression model of sqft_living to price 
# lets split data into training and tests


# In[ ]:


train_data, test_data = sales.random_split(.8, seed=0) # 80% is for training and 20% is for testing


# In[2]:


# Build a regression model 


# In[ ]:


sqft_model = graphlab.linear_regression.create(train_data, target='price', 
                                               feature=["sqft_livig"]) # target mean what we are trying to predict


# In[ ]:


# evaluting error of our model


# In[ ]:


print test_data['price'].mean() 
# test our model on test data


# In[ ]:


sqft_model.evaluate(test_data)


# In[3]:


# now lets look at some prediction from our data


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.plot(test_data['sqft_living'], test_data['price'], '.', # plot a scatter plot
        test_data['sqft_living'], sqft_model.predict(test_data), '-') # our prediction (abline)


# In[ ]:


sqft_model.get('coefficients') # also knwon as weights
# based on the model, on average a sqft cost 280 dollars


# In[ ]:


# NOW LETS EXPLORE OTHER FEATURES 


# In[ ]:


features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode'] # zipcode also known as postalcode


# In[ ]:


sales[features].show()


# In[ ]:


sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# In[ ]:


# Build another regression model with more features


# In[ ]:


feature_model = graphlab.linear_regression.create(train_data, target='price', features=features)


# In[ ]:


# compare both models to see how good they are

sqft_model.evaluate(test_data)
feature_model.evaluate(test_data)


# In[ ]:


# apply our models to predict house prices


# In[ ]:


house1 = sales[sales['id']=='5309101200'] # selcting a specific house


# In[ ]:


# let see the house picture
cimg src='house-5309101200.jpg'


# In[ ]:


house1['price']


# In[ ]:


# lets see what the model predict
sqft_model.predict(house1)


# In[ ]:


features_model.predict(house1)


# In[ ]:


# Lets predict a fancy house


# In[ ]:


house2 = sales[sales['id'] =='1925069082']


# In[ ]:


house2['price']


# In[ ]:


cimg src='house-1925069082' #The picture is in my directory


# In[ ]:


sqft_model.predict(house2) 


# In[ ]:


features_model.predict(house2) # a house with more features such as goog location, this model is good to use


# In[ ]:


# lets look at more fancier house


# In[4]:


bill_gates = {'bedrooms': [8],
             'bathroom': [25],
             'sqft_living': [5000]} #etc


# In[ ]:


features_model.predict(graphlab.Sframe(bill_gates)) # convert dict to sframe dict


# In[ ]:




