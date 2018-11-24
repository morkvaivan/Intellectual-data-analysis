import pandas as pd
data = pd.read_csv('titanic_train.csv', sep=',')

data.age[data.age.isnull()] = data.age.median()

MaxPassEmbarked = data.groupby('embarked').count()['row.names']
data.embarked[data.embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

data = data.drop(['name','ticket','room', 'home.dest', 'boat'],axis=1)

numberOfAllWoman = data['row.names'][data.sex == 'female'].count()

sumOfAllWomanAges = data['age'][data.sex == 'female'].sum()

result = sumOfAllWomanAges / numberOfAllWoman

print("\n average age of female passengers: \n\n" + str( result ))