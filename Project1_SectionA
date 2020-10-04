
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("BreadBasket_DMS.csv")

data = data[data.Item != 'NONE']
plt.figure(4, figsize=(7, 7))
print("Frequency of top 10 selling items...")
print(data['Item'].value_counts().head(10)) # Question 1
plt.show()
data = data.drop(['Time'], axis=1)
data.Date = pd.to_datetime(data.Date)      # Question 2 and 3
data = data[data.Date > '2016-12-31']      # Question 2 and 3

print("      Five most popular bakery items in 2017...")
print(data.sort_values("Transaction", ascending=False).head(5))

# Quesiton 2:

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data.Date = pd.to_datetime(data.Date)
data = data[data.Date < '2016-12-31']
data['Item'].value_counts().head(5).plot(kind='bar', color='blue')
print("Five most popular bakery items in 2017.  ")
print(data['Item'].value_counts().head(5))
plt.show()

# Question 3:

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data.Date = pd.to_datetime(data.Date)
data = data[data.Date > '2016-12-31']
data['Item'].value_counts().head(5).plot(kind='bar', color='blue')
print("Five most popular bakery items in 2017.  ")
print(data['Item'].value_counts().head(5))
plt.show()

# Question 4

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['weekday'] = data['datetime'].dt.weekday
data.Date = pd.to_datetime(data.Date)
data = data[data.weekday == 0]
data['Item'].value_counts().head(5).plot(kind='bar', color='blue')
print("Five most popular bakery items sold on Monday.  ")
print(data['Item'].value_counts().head(5))
plt.show()

#Question 5
print("Number of Transactions per Month...")
data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['month'] = data['datetime'].dt.month
weeklySales = data['month'].value_counts()
weeklySales = weeklySales.sort_index()
getSales = weeklySales.rename(index={1:'January',2:'Febrary', 3:'March',4: 'April', 5:'May',
                                     6:'June', 7: 'July',8:'August', 9:'September',10: 'October', 11:'November', 12:'December'})

data['month'].value_counts().sort_index().plot(kind='bar', color='blue')
print(getSales)
plt.show()


#Question 6
print("Number of Transactions per weekday...")
data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['weekday'] = data['datetime'].dt.weekday
weeklySales = data['weekday'].value_counts()
weeklySales = weeklySales.sort_index()
getSales = weeklySales.rename(index={0:'Monday',1:'Tuesday',
                        2:'Wednesday',3: 'Thursday', 4:'Friday', 5:'Saturday', 6: 'Sunday'})

data['weekday'].value_counts().sort_index().plot(kind='bar', color='blue')
print(getSales)
plt.show()

#Question 7

print("Number of Transactions per hours...")
data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['hour'] = data['datetime'].dt.hour
hourlysales = data['hour'].value_counts()
hourlysales = hourlysales.sort_index()

data['hour'].value_counts().sort_index().plot(kind='bar', color='blue')
print(hourlysales)
plt.show()


#Question 9

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['year'] = data['datetime'].dt.year
data['day'] = data['datetime'].dt.day
data.Date = pd.to_datetime(data.Date)
data = data[data.Date > '2016-12-31']
data = data[data.Item == "Coffee"]
data = data.drop(['Time'], axis=1)
print("Five most Coffee Sales day in 2017...")
data['Date'].value_counts().head(5).plot(kind='bar', color='blue')
print(data["Date"].value_counts().head(5))
plt.show()

# Quesiton 10

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['weekday'] = data['datetime'].dt.weekday
data = data[data.Item == "Bread"]
print(data['weekday'].value_counts())
data['weekday'].value_counts().sort_index().plot(kind='bar', color='blue')
plt.ylabel("Sales")
plt.title("Weekday Bread Sales")
plt.xlabel("Week ('0 = Sunday')")
plt.show()

