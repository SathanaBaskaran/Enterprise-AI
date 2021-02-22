
"""
## Importing libraries :
"""
import numpy as np
import pandas as pd
import scipy
import itertools
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import wordcloud
# from openpyxl import load_workbook

"""
## Loading the Dataset :
"""
def segmentation():
    data = pd.read_excel('D:\ENTERPRISE_AI\env\datas\Retail1.xls', dtype={'StockCode':str})

    data.head(3)
    """
    ## Data Preprocessing :
    """
    # Checking for null values.
    info = pd.DataFrame(data=data.isnull().sum()).T.rename(index={0:'Null values'})
    info = info.append(pd.DataFrame(data=data.isnull().sum()/data.shape[0] * 100).T.rename(index={0:'% Null values'}))
    """
    > Since we dont have CustomerID for 25% of points we will remove them as we cannot give 
    them any arbitrary ID.
    """
    # Removing null values
    data.dropna(axis=0, subset = ['CustomerID'], inplace=True)
    info = pd.DataFrame(data=data.isnull().sum()).T.rename(index={0:'Null values'})
    info = info.append(pd.DataFrame(data=data.isnull().sum()/data.shape[0] * 100).T.rename(index={0:'% Null values'}))
    # Checking for Duplicates :
    data.duplicated().sum()
    # Removing duplicate entries :
    data.drop_duplicates(inplace=True)
    data.duplicated().sum()
    """
    ## Exploratory Data Analysis :
    """
    Country_classification = data.groupby(['Country']).groups.keys()
    Country_classification_count = dict(data.groupby(['Country'])['CustomerID'].count())
    print("COUNTRY CASSIFICATION\n")
    # print(data.groupby(['Country']).groups.keys(), data.groupby(['Country'])['CustomerID'].count())
    info = pd.DataFrame(data = data.groupby(['Country'])['InvoiceNo'].nunique(), index=data.groupby(['Country']).groups.keys()).T

    # StockCode Feature ->
    # We will see how many different products were sold in the year data was collected.
    print("Number of products sold this year")
    product_sold = len(data['StockCode'].value_counts())
    print(len(data['StockCode'].value_counts()))
    # Transanction feature
    # different number of transanctions.
    print("Number of Transactions:")
    no_of_transactions =len(data['InvoiceNo'].value_counts())
    print(len(data['InvoiceNo'].value_counts()))
    # Transanction feature
    # Number of different Customers.
    print("Number of different Customers:")
    no_of_diff_customers = len(data['CustomerID'].value_counts())
    print(len(data['CustomerID'].value_counts()))
    pd.DataFrame({'products':len(data['StockCode'].value_counts()),
                'transanctions':len(data['InvoiceNo'].value_counts()),
                'Customers':len(data['CustomerID'].value_counts())},
                index = ['Quantity'])
    
    """
    ##### Checking the number of items bought in a single transanctions :
    """
    df = data.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
    df = df.rename(columns = {'InvoiceDate':'Number of products'})
    df[:10].sort_values('CustomerID')
    """
    > There are customers who purchase only 1 item per transanction and others who purchase many items per transanction.
    Also there are some orders which were cancelled they are marked with 'C' in the beginning.
    """
    """
    ##### Counting number of cancelled transanctions :
    """
    df['orders cancelled'] = df['InvoiceNo'].apply(lambda x: int('C' in str(x)))
    df.head()
    
    # Printing number of orders cancelled ->
    order_cancelled = df['orders cancelled'].sum()
    total_orders = df.shape[0] 
    percent_ofcancelledOrder =  (df['orders cancelled'].sum()/ df.shape[0] * 100)
    print("Number of orders cancelled {}/{}estimated percentage({:.2f}%)".format(df['orders cancelled'].sum(), df.shape[0], df['orders cancelled'].sum()/ df.shape[0] * 100))
    """
    ##### Handling Cancelled Values :
    """
    # Looking at cancelled transanctions in original data.
    data.sort_values('CustomerID')[:5]
    """
    ##### Checking for discounted products :
    """
    df = data[data['Description'] == 'Discount']
    """
    ##### Checking whether every order that has been cancelled has a counterpart :
    """
    df = data[(data['Quantity']<0) & (data['Description']!='Discount')][['CustomerID','Quantity','StockCode','Description','UnitPrice']]
    for index, col in df.iterrows():
        if data[(data['CustomerID'] == col[0]) & (data['Quantity'] == -col[1]) & (data['Description'] == col[2])].shape[0] == 0:
            print(index, df.loc[index])
            print("There are some transanctions for which counterpart does not exist")
            break
    """
    ##### Removing cancelled orders :
    """
    df_cleaned = data.copy(deep=True)
    df_cleaned['QuatityCancelled'] = 0
    entry_to_remove = []; doubtfull_entry = []
    for index, col in data.iterrows():
        if(col['Quantity'] > 0)or(col['Description']=='Discount'):continue
        df_test = data[(data['CustomerID']==col['CustomerID'])&(data['StockCode']==col['StockCode'])&
                    (data['InvoiceDate']<col['InvoiceDate'])&(data['Quantity']>0)].copy()
        # Order cancelled without counterpart, these are doubtful as they maybe errors 
        # or maybe orders were placed before data given
        if(df_test.shape[0] == 0):
            doubtfull_entry.append(index)
        # Cancellation with single counterpart
        elif(df_test.shape[0] == 1):
            index_order = df_test.index[0]
            df_cleaned.loc[index_order, 'QuantityCancelled'] = -col['Quantity']
            entry_to_remove.append(index)
        # Various counterpart exists for orders
        elif(df_test.shape[0] > 1):
            df_test.sort_index(axis = 0, ascending=False, inplace=True)
            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']:continue
                df_cleaned.loc[ind, 'QuantityCancelled'] = -col['Quantity']
                entry_to_remove.append(index)
                break
    print("Entry to remove {}".format(len(entry_to_remove)))
    print("Doubtfull Entry {}".format(len(doubtfull_entry)))
    # Deleting these entries :
    df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
    df_cleaned.drop(doubtfull_entry, axis=0, inplace=True)
    """
    ##### We will now see the StockCode feature especially the discounted items:
    """
    list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex = True)]['StockCode'].unique()
    print(list_special_codes)
    for code in list_special_codes:
        print("{:<17} -> {:<35}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].values[0]))
    df_cleaned['QuantityCancelled'] = np.nan_to_num(df_cleaned['QuantityCancelled'])
    df_cleaned.head()
    """
    > We see that the same transanction is duplicated for every different item in the dataset.
    """
    """
    ##### Getting total data feature :
    """
    df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCancelled'])
    print(df_cleaned.sort_values('CustomerID'))
    """
    ##### Now we sum the individual orders and group them on the basis of invoice number 
    # to remove the problem of duplicate rows for same order :
    """
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket_price = temp.rename(columns = {'TotalPrice': 'Basket Price'})
    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis = 1, inplace=True)
    basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
    basket_price = basket_price[basket_price['Basket Price'] > 0]
    basket_price.sort_values('CustomerID')[:6]
    """
    ##### Plotting the purchases made :
    """

    price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
    count_price = []
    for i,price in enumerate(price_range):
        if i==0:continue
        val = basket_price[(basket_price['Basket Price'] < price)&
                        (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
        count_price.append(val)
        print(val)
    """
    ##### Analyzing product Description :
    """
    is_noun = lambda pos:pos[:2] == 'NN'
    def keywords_inventory(dataframe, colonne = 'Description'):
        import nltk
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots = dict()
        keywords_select = dict()
        category_keys = []
        count_keywords = dict()
        icount = 0
        
        for s in dataframe[colonne]:
            if pd.isnull(s): continue
            lines = s.lower()
            tokenized = nltk.word_tokenize(lines)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
            
            for t in nouns:
                t = t.lower() ; racine = stemmer.stem(t)   
                if racine in keywords_roots:
                    keywords_roots[racine].add(t)
                    count_keywords[racine] += 1
                else:
                    keywords_roots[racine] = {t}
                    count_keywords[racine] = 1
            
        
        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        clef = k ; min_length = len(k)
                
                category_keys.append(clef)
                keywords_select[s] = clef
            
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]
                
        print("Number of keywords in the variable '{}': {}".format(colonne, len(category_keys)))
        return category_keys, keywords_roots, keywords_select, count_keywords

    df_produits = pd.DataFrame(data['Description'].unique()).rename(columns = {0:"Description"})
    keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)
    # Plotting keywords vs frequency graph :
    list_products = []
    # Preserving important words :
    list_products = []
    for k, v in count_keywords.items():
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word)<3 or v<13: continue
        list_products.append([word, v])
        
    list_products.sort(key = lambda x:x[1], reverse=True)
    print("Number of preserved words : ", len(list_products))
    """
    ##### Descrining every product in terms of words present in the description :
    1. We will only use the preserved words, this is just like Binary Bag of Words<br>
    2. We need to convert this into a product matrix with products as rows and different words as columns. A cell contains a 1 if a particular product has that word in its description else it contains 0.
    3. We will use this matrix to categorize the products.
    4. We will add a mean price feature so that the groups are balanced.
    """
    threshold = [0, 1, 2, 3, 5, 10]
    # Getting the description.
    liste_produits = df_cleaned['Description'].unique()
    # Creating the product and word matrix.
    X = pd.DataFrame()
    for key, occurence in list_products:
        X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))
        

    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold) - 1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i], threshold[i+1])
            
        label_col.append(col)
        X.loc[:, col] = 0
        
    for i, prod in enumerate(liste_produits):
        prix = df_cleaned[df_cleaned['Description'] == prod]['UnitPrice'].mean()
        j = 0
        
        while prix > threshold[j]:
            j += 1
            if j == len(threshold):
                break
        X.loc[i, label_col[j-1]] = 1
    print("{:<8} {:<20} \n".format('range', 'number of products') + 20*'-')
    # ProductsInrange = {}
    ProductsInrange=[]
    for i in range(len(threshold)):
        if i == len(threshold)-1:
            col = '.>{}'.format(threshold[i])
            # #print(col)
        else:
            col = '{}<.<{}'.format(threshold[i],threshold[i+1])
           # print(col)
        # ProductsInrange[str(col)] = X.loc[:, col].sum()
        ProductsInrange.append(X.loc[:, col].sum())
        print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))
    """
    ### Clustering :
    # 1. KMEANS.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    matrix = X.to_numpy()
    # Using optimal number of clusters using hyperparameter tuning:
    for n_clusters in range(3, 10):
        kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init = 30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        sil_avg = silhouette_score(matrix, clusters)
        print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)
    # Choosing number of clusters as 5:
    # Trying Improving the silhouette_score :
    n_clusters = 5
    sil_avg = -1
    while sil_avg < 0.145:
        kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        sil_avg = silhouette_score(matrix, clusters)
        print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)
    # Printing number of elements in each cluster :
    pd.Series(clusters).value_counts()
    # print(pd.Series(clusters).value_counts())

    """
    #### Analyzing the 5 clusters :
    """
segmentedVals = {"CountryCount": Country_classification_count,"product_sold": product_sold,"totalTransactions": no_of_transactions,"NoOfdiffCustomers": no_of_diff_customers,"totalOrders": total_orders,"CancelledOrders": order_cancelled,"perOfCancelledorder":percent_ofcancelledOrder,"ProductsInrange":ProductsInrange} return segmentedVals
   
"""
    ##### Analysis using wordcloud:
    > Checking which words are most common in the clusters.
    # """
liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurance) in list_products]

occurance = [dict() for _ in range(n_clusters)]
# Creating data for printing word cloud.
for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurance[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
"""
## Dimensionality Reduction:
> PCA
"""
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(matrix)
pca_samples = pca.transform(matrix)
"""
## Generating Customer Segments/Categories :
 already generated product categories are used to creat a new feature which tells to which category the product belongs to.
"""
corresp = dict()
for key, val in zip(liste_produits, clusters):
    corresp[key] = val
    
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)
df_cleaned[['InvoiceNo', 'Description', 'categ_product']][:10]

# Creating 5 new features that will contain the amount in a single transanction on 
# different categories of product.
for i in range(5):
    col = 'categ_{}'.format(i)
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCancelled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
    
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']][:10]
"""
##### A single order is split into multiple entries we will basket them :
"""
# sum of purchases by user and order.
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index = False)['TotalPrice'].sum()
basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})
# percentage spent on each product category 
for i in range(5):
    col = "categ_{}".format(i)
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index = False)[col].sum()
    basket_price.loc[:col] = temp
# Dates of the order.
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index = False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
# Selecting entries with basket price > 0.
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending=True)[:5]

basket_price['InvoiceDate'].min()

basket_price['InvoiceDate'].max()
"""
##### Time Based Splitting :
"""
import datetime
from datetime import datetime

date = '2011-10-01'
date = datetime.strptime(date, '%Y-%m-%d')
set_entrainment = basket_price[basket_price['InvoiceDate'] < date]
set_test = basket_price[basket_price['InvoiceDate'] >= date]
basket_price = set_entrainment.copy(deep = True)
"""
##### Grouping Orders :
> We will get info about every customer on how much do they purchase, total number of orders. etc
"""
transanctions_per_user = basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count', 'min', 'max', 'mean', 'sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    testing=(basket_price.groupby(by=['CustomerID'])[col].sum())
    transanctions_per_user.loc[:col] =testing/(transanctions_per_user['sum'] * 100)
    
transanctions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transanctions_per_user.sort_values('CustomerID', ascending = True)[:5]
# Generating two new variables - days since first puchase and days since last purchase.
last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transanctions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transanctions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']

n1 = transanctions_per_user[transanctions_per_user['count'] == 1].shape[0]
n2 = transanctions_per_user.shape[0]
print("No. of Customers with single purchase : {:<2}/{:<5} ({:<2.2f}%)".format(n1, n2, n1/n2*100))
"""
##### Building Customer Segments :
"""
list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
selected_customers = transanctions_per_user.copy(deep=True)
matrix = selected_customers[list_cols].as_matrix()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(matrix)
print("Variable Mean Values: \n" + 90*'-' + '\n', scaler.mean_)
scaled_matrix = scaler.transform(matrix)

pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)

# Using optimal number of clusters using hyperparameter tuning:
for n_clusters in range(3, 21):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init = 30)
    kmeans.fit(scaled_matrix)
    clusters = kmeans.predict(scaled_matrix)
    sil_avg = silhouette_score(scaled_matrix, clusters)
    print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)
# Choosing number of clusters as 10:
# Trying Improving the silhouette_score :
n_clusters = 10
sil_avg =-1
while sil_avg < 0.208:
    kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
    kmeans.fit(scaled_matrix)
    clusters = kmeans.predict(scaled_matrix)
    sil_avg = silhouette_score(scaled_matrix, clusters)
    print("For n_clusters : ", n_clusters, "The average silhouette_score is : ", sil_avg)

n_clusters = 10
kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print("Silhouette Score : {:<.3f}".format(silhouette_avg))

# Looking at clusters :
pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns=['Number of Clients']).T
"""
##### Now we need to learn the habits of the customers to do that we will add the variables that define a cluster to which each customer belong:
"""
selected_customers.loc[:, 'cluster'] = clusters_clients

merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
    
merged_df.drop('CustomerID', axis = 1, inplace = True)
print('Number of customers : ', merged_df['size'].sum())

merged_df = merged_df.sort_values('sum')
# Reorganizing the content of the dataframe.
liste_index = []
for i in range(5):
    column = 'categ_{}'.iloc(i)
    liste_index.append(merged_df[merged_df[column] > 45].index.values[0])
    
liste_index_reordered = liste_index
liste_index_reordered += [s for s in merged_df.index if s not in liste_index]

merged_df = merged_df.reindex(index = liste_index_reordered)
merged_df = merged_df.reset_index(drop = False)
"""
## Classifying the Customers :
"""
selected_customers = pd.read_csv('Retail1.csv')
merged_df = pd.read_csv('Retail1.csv')
"""
##### Defining Helper Functions :
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
class Class_Fit(object):
    def __init__(self, clf, params = None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()
            
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    
    def predict(self, x):
        return self.clf.predict(x)
    
    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator = self.clf, param_grid = parameters, cv = Kfold)
        
    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)
        
    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        print("Precision: {:.2f} %".format(100 * accuracy_score(Y, self.predictions)))

selected_customers.head()

columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
X = selected_customers[columns]
Y = selected_customers['cluster']

"""
##### Train, Test Splitting :
"""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

"""
### Training Models :
"""

from sklearn.svm import LinearSVC

svc = Class_Fit(clf=LinearSVC)
svc.grid_search(parameters = [{'C':np.logspace(-2,2,10)}], Kfold = 5)

svc.grid_fit(X=X_train, Y=Y_train)

svc.grid_predict(X_test, Y_test)

from sklearn.metrics import confusion_matrix

## Code from sklearn documentation.
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

"""
#### Logistics Regression :
"""
from sklearn.linear_model import LogisticRegression
lr = Class_Fit(clf = LogisticRegression)
lr.grid_search(parameters = [{'C':np.logspace(-1,2,10)}], Kfold = 5)
lr.grid_fit(X_train, Y_train)
lr.grid_predict(X_test, Y_test)

"""
#### K-Nearest Neighbours :
"""
from sklearn.neighbors import KNeighborsClassifier
knn = Class_Fit(clf = KNeighborsClassifier)
knn.grid_search(parameters = [{'n_neighbors':np.arange(1,50,1)}], Kfold = 5)
knn.grid_fit(X_train, Y_train)
knn.grid_predict(X_test, Y_test)
"""
#### Decision Trees :
"""
from sklearn.tree import DecisionTreeClassifier
tr = Class_Fit(clf = DecisionTreeClassifier)
tr.grid_search(parameters = [{'criterion':['entropy', 'gini'], 'max_features':['sqrt', 'log2']}], Kfold = 5)
tr.grid_fit(X_train, Y_train)
tr.grid_predict(X_test, Y_test)
"""
#### Random Forests:
"""
from sklearn.ensemble import RandomForestClassifier

rf = Class_Fit(clf = RandomForestClassifier)
rf.grid_search(parameters = [{'criterion':['entropy', 'gini'], 
                              'max_features':['sqrt', 'log2'], 'n_estimators':[20, 40, 60, 80, 100]}], Kfold = 5)
rf.grid_fit(X_train, Y_train)
rf.grid_predict(X_test, Y_test)

from sklearn.ensemble import AdaBoostClassifier

ada = Class_Fit(clf = AdaBoostClassifier)
ada.grid_search(parameters = [{'n_estimators':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}], Kfold = 5)
ada.grid_fit(X_train, Y_train)
ada.grid_predict(X_test, Y_test)
"""
#### Gradient Boosted Decision Trees :
"""
import xgboost

gbdt = Class_Fit(clf = xgboost.XGBClassifier)
gbdt.grid_search(parameters = [{'n_estimators':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}], Kfold = 5)
gbdt.grid_fit(X_train, Y_train)
gbdt.grid_predict(X_test, Y_test)
"""
#### Voting Classifier :
"""

rf_best = RandomForestClassifier(**rf.grid.best_params_)
gbdt_best = xgboost.XGBClassifier(**gbdt.grid.best_params_)
svc_best = LinearSVC(**svc.grid.best_params_)
tr_best = DecisionTreeClassifier(**tr.grid.best_params_)
knn_best = KNeighborsClassifier(**knn.grid.best_params_)
lr_best = LogisticRegression(**lr.grid.best_params_)

from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators=[('rf', rf_best), ('gb', gbdt_best), ('knn', knn_best), ('lr', lr_best)])

votingC = votingC.fit(X_train, Y_train)

predictions = votingC.predict(X_test)

print("Precision : {:.2f}%".format(100 * accuracy_score(Y_test, predictions)))
"""
### Testing the model :
"""
basket_price = set_test.copy(deep=True)
transanctions_per_user = basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count', 'min', 'max', 'mean', 'sum'])

for i in range(5):
    col = 'categ_{}'.format(i)
    transanctions_per_user.loc[:, col] = basket_price.groupby(by=['CustomerID'])[col].sum() / transanctions_per_user['sum'] * 100
    
transanctions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()

transanctions_per_user['count'] = 5 * transanctions_per_user['count']
transanctions_per_user['sum'] = transanctions_per_user['count'] * transanctions_per_user['mean']

transanctions_per_user.sort_values('CustomerID', ascending = True)[:5]

list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
matrix_test = transanctions_per_user[list_cols].as_matrix()
scaled_test_matrix = scaler.transform(matrix_test)

Y = kmeans.predict(scaled_test_matrix)
columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4' ]
X = transanctions_per_user[columns]
predictions = votingC.predict(X)

print("Precision : {:.2f}%".format(100 * accuracy_score(Y, predictions)))
     


