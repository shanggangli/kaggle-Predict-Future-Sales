import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb
from sklearn.metrics import mean_squared_error #RMSE
from math import sqrt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
Traindata=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Predict Future Sales\sales_train.csv'))
#print(Traindata.info())
Traindata['date']=pd.to_datetime(Traindata['date'])
Traindata['year']=Traindata.date.dt.year
Traindata['month'] = Traindata.date.dt.month
Traindata['day'] = Traindata.date.dt.day

# 不同年份的总销售额
'''sale_and_year=Traindata[['item_cnt_day','year']].groupby(['year']).count().plot(kind='bar')
plt.title('sale in years')
plt.show()

# 不同年份的月销售额
# 2013
Traindata_2013=Traindata[Traindata['year']==2013]
sale_and_month=Traindata_2013[['item_cnt_day','month']].groupby(['month']).count().plot(kind='bar')
plt.title('12months’ sale in 2013')
plt.show()
# 2014
Traindata_2014=Traindata[Traindata['year']==2014]
sale_and_month=Traindata_2014[['item_cnt_day','month']].groupby(['month']).count().plot(kind='bar')
plt.title('12months’ sale in 2014')
plt.show()
# 2015
Traindata_2015=Traindata[Traindata['year']==2015]
sale_and_month=Traindata_2015[['item_cnt_day','month']].groupby(['month']).count().plot(kind='bar')
plt.title('12months’ sale in 2015')
plt.show()


# 2013年不同商店的月销售额
Traindata_2013=Traindata[Traindata['year']==2013]
shop_and_sale=Traindata_2013[['shop_id','month','item_cnt_day']].groupby(['shop_id','month']).count().reset_index()
#print(shop_and_sale)
shop_and_sale_pivot=shop_and_sale.pivot(index='shop_id',columns='month',values='item_cnt_day')
#print(shop_and_sale_pivot)
shop_and_sale_pivot.plot(kind='bar')
plt.show()

# 2014
Traindata_2014=Traindata[Traindata['year']==2014]
shop_and_sale=Traindata_2014[['shop_id','month','item_cnt_day']].groupby(['shop_id','month']).count().reset_index()
#print(shop_and_sale)
shop_and_sale_pivot=shop_and_sale.pivot(index='shop_id',columns='month',values='item_cnt_day')
#print(shop_and_sale_pivot)
shop_and_sale_pivot.plot(kind='bar')
plt.show()

# 2015
Traindata_2015=Traindata[Traindata['year']==2015]
shop_and_sale=Traindata_2015[['shop_id','month','item_cnt_day']].groupby(['shop_id','month']).count().reset_index()
#print(shop_and_sale)
shop_and_sale_pivot=shop_and_sale.pivot(index='shop_id',columns='month',values='item_cnt_day')
#print(shop_and_sale_pivot)
shop_and_sale_pivot.plot(kind='bar')
plt.show()'''

# 不同商品类型的总销售量
items=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Predict Future Sales\items.csv'))
items=items.drop(['item_name'],axis=1,inplace=False)
items=items.set_index('item_id').to_dict(orient='dict')
Traindata['category']=Traindata['item_id'].map(items['item_category_id'])
'''sale_and_category=Traindata[['item_cnt_day','category']].groupby(['category']).count().plot(kind='bar')
plt.title('sale in categories')
plt.show()'''
# 销售量超过10W的商品类型
'''sale_and_category=Traindata[['item_cnt_day','category']].groupby(['category']).count()
top=sale_and_category[sale_and_category['item_cnt_day']>=100000].plot(kind='bar')
plt.title('sale over 10w in categories')
plt.show()'''
#print(top)

# 不同商品类型的平均价格
'''prices_and_category=Traindata[['category','item_price']].groupby(['category']).mean().plot(kind='bar')
plt.title('mean prices in categories')
plt.show()'''

'''fig,axes = plt.subplots()
prices_and_category.boxplot(by='category',ax=axes)
temp=Traindata.item_price[Traindata['category']==12]
print(temp.describe())'''

# 数据处理
#Traindata=Traindata.drop(Traindata[Traindata['item_cnt_day']])
'''pd.DataFrame(Traindata['item_price']).boxplot()
plt.show()'''
# 处理价格
#print(pd.DataFrame(Traindata[Traindata.item_price>=30000]).info())
Traindata=Traindata.drop(Traindata[Traindata.item_price<=0].index | Traindata[Traindata.item_price>=30000].index)
'''pd.DataFrame(Traindata['item_price']).boxplot()
plt.show()'''

# 处理销售额
#print(pd.DataFrame(Traindata[Traindata['item_cnt_day']<-10]).info())
Traindata=Traindata.drop(Traindata[Traindata.item_cnt_day<-10].index | Traindata[Traindata.item_cnt_day>=500].index)
'''pd.DataFrame(Traindata['item_cnt_day']).boxplot()
plt.show()'''
# 不同商店不同产品类型价格的平均值
mean_price=pd.pivot_table(Traindata,index=['shop_id','category'],values=['item_price'],aggfunc=[np.mean],fill_value=0).reset_index()
mean_price.columns=['shop_id','category','mean']

## 销售额
# 不同商店的月销售额均值
month_sale=pd.pivot_table(Traindata,index=['shop_id','date_block_num'],values=['item_cnt_day'],aggfunc=[np.mean],fill_value=0) .reset_index()
month_sale.columns=['shop_id','month_id','saleOn_month']

# 不同类型商品的月销售额均值
month_category=pd.pivot_table(Traindata,index=['category','date_block_num'],values=['item_cnt_day'],aggfunc=[np.mean],fill_value=0) .reset_index()
month_category.columns=['category','month_id','month_category']

# 不同商店的年总销售额
year_saleonShop=pd.pivot_table(Traindata,index=['shop_id','year'],values=['item_cnt_day'],aggfunc=[np.mean],fill_value=0) .reset_index()
year_saleonShop.columns=['shop_id','year','year_saleonShop']

# 不同类型商品的年总销售额
year_saleonCategory=pd.pivot_table(Traindata,index=['category','year'],values=['item_cnt_day'],aggfunc=[np.mean],fill_value=0) .reset_index()
year_saleonCategory.columns=['category','year','year_saleonCategory']

# 不同商品的月销售额均值
month_item=pd.pivot_table(Traindata,index=['item_id','date_block_num'],values=['item_cnt_day'],aggfunc=[np.mean],fill_value=0) .reset_index()
month_item.columns=['item_id','month_id','month_item']

# 不同商品在不同店铺的月销售额均值
month_itemandshop=pd.pivot_table(Traindata,index=['item_id','shop_id','date_block_num'],values=['item_cnt_day'],aggfunc=[np.sum],fill_value=0) .reset_index()
month_itemandshop.columns=['item_id','shop_id','month_id','month_itemandshop']

#
train_data1= pd.pivot_table(Traindata,index=['shop_id','item_id','date_block_num','year','category'], values=['item_cnt_day'], aggfunc=[np.sum],fill_value=0).reset_index()
train_data1.columns=['shop_id','item_id','month_id','year','category','item_cnt_month']

combined_data=pd.merge(train_data1,mean_price,on=['shop_id','category'])
combined_data=pd.merge(combined_data,month_sale,on=['shop_id','month_id'])
combined_data=pd.merge(combined_data,month_category,on=['category','month_id'])
combined_data=pd.merge(combined_data,year_saleonShop,on=['shop_id','year'])
combined_data=pd.merge(combined_data,year_saleonCategory,on=['category','year'])
combined_data=pd.merge(combined_data,month_item,on=['item_id','month_id'])
combined_data=pd.merge(combined_data,month_itemandshop,on=['item_id','shop_id','month_id'])
#print(combined_data.info())
#print(train_data1.info())
'''from scipy.stats import pearsonr
print(pearsonr(combined_data['item_cnt_month'],combined_data['mean']))'''

# 做一个baseline
X=combined_data[['shop_id','item_id','month_id','year','category','mean','saleOn_month','month_category','year_saleonShop','year_saleonCategory','month_item','month_itemandshop']]
y=combined_data['item_cnt_month']
#print(mutual_info_regression(X,y))
'''selector=SelectKBest(score_func=mutual_info_regression,k=5)
selector.fit(X,y)
print(selector.scores_)
print(selector.pvalues_)'''
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,train_size=0.8)
print(",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
print(",训练数据标签:",y_train.shape,
     ',测试数据标签:',y_test.shape )

model = xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2);
xgb_starttime=time.time()
model.fit(X_train, y_train)
costtime=time.time()-xgb_starttime
pred=model.predict(X_test)
#print(pred)
rms=sqrt(mean_squared_error(y_test,pred))
print(rms)
print('花费时间:',costtime)

'''fpr,tpr=roc_curve(y_test,pred)
print('AUC:',auc(fpr,tpr))
print('花费时间:',costtime)'''

# Test
Testdata=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\kaggle\Predict Future Sales\test.csv'))
Testdata['year']=2015
Testdata['month']=34
Testdata=pd.merge(Testdata,items,on=['item_id'])
#print(Testdata.info())

