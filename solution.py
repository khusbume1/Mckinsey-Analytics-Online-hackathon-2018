
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing.imputation import Imputer
train = pd.read_csv('C:/Users/KHUSHBU/OneDrive - Mahidol University/mckinseydata/train_ZoGVYWq.csv', sep=',')
test = pd.read_csv('C:/Users/KHUSHBU/OneDrive - Mahidol University/mckinseydata/test_66516Ee.csv', sep=',' )
model = GaussianNB()
residence_area_type_p = {'Rural': 0,'Urban': 1}
train.residence_area_type = [residence_area_type_p[item] for item in train.residence_area_type]
sourcing_channel_p = {'A':1,'B':2, 'C':3,'D':4,'E':4,'F':5 }
train.sourcing_channel = [sourcing_channel_p[item] for item in train.sourcing_channel]
columns_to_impute = ['application_underwriting_score', 'Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late']
train[columns_to_impute] = Imputer().fit_transform(train[columns_to_impute])
label = train.renewal
p = train.drop(['renewal'],axis=1)
#remove the insignificant variables (obtained from the result of significance test done separately)
#p = train.drop(['residence_area_type'],axis=1)
#p = train.drop(['Income'],axis=1)
model.fit(p, label)
## make predictions
test.residence_area_type = [residence_area_type_p[item] for item in test.residence_area_type]
test.sourcing_channel = [sourcing_channel_p[item] for item in test.sourcing_channel]
test[columns_to_impute] = Imputer().fit_transform(test[columns_to_impute])
#test.drop(['residence_area_type'],axis=1)
#test.drop(['Income'],axis=1)
predicted = model.predict_proba(test)
df = pd.DataFrame(predicted)
df['effort_in_hours'] = np.where(df.iloc[:,1]>=0.85, 5, 3)
f = df['effort_in_hours']*-1
df['incentive'] = pd.Series(-400*(np.log((f/10)+1))).astype(float)
df['percent_improvement_in_renewal_prob'] = 20*(1-np.exp(f/5))
df.columns.values[1] = 'Pbenchmark'
df['deltaP']= df['percent_improvement_in_renewal_prob'] * df['Pbenchmark']
m = ((df['Pbenchmark']+df['deltaP'])*test['premium'])
df['netrevenue'] = m-df['incentive']
df['id'] = test['id']
submission = df[['id', 'Pbenchmark', 'incentive']].copy()
submission.columns.values[1]='renewal'
submission.columns.values[2]='incentives'
submission.to_csv("submission.csv", sep=',',encoding='utf-8')
