
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv('BreadBasket_DMS.csv')
transaction_list = []
for i in df['Transaction'].unique():
    tlist = list(set(df[df['Transaction'] == i]['Item']))
    if len(tlist) > 0:
        transaction_list.append(tlist)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

getCountList = []
te = TransactionEncoder()
te_ary = te.fit(transaction_list).transform(transaction_list)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

# Question 4

frequent_itemsets = apriori(df2, min_support=0.1, use_colnames=True)

# Confidence Level = 90%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
getCountList.append(len(rules))
print("Rules generated at 90% Confidence:  " + str(len(rules)))

# Confidence Level = 80%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)
getCountList.append(len(rules))
print("Rules generated at 80% Confidence:  " + str(len(rules)))

# Confidence Level = 70%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
getCountList.append(len(rules))
print("Rules generated at 70% Confidence:  " + str(len(rules)))

# Confidence Level = 60%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
getCountList.append(len(rules))
print("Rules generated at 60% Confidence:  " + str(len(rules)))

# Confidence Level = 50%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
getCountList.append(len(rules))
print("Rules generated at 50% Confidence:  " +str(len(rules)))

# Confidence Level = 40%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
getCountList.append(len(rules))
print("Rules generated at 40% Confidence:  " + str(len(rules)))

# Confidence Level = 30%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
getCountList.append(len(rules))
print("Rules generated at 30% Confidence:  " + str(len(rules)))

# Confidence Level = 20%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)
getCountList.append(len(rules))
print("Rules generated at 20% Confidence:  " + str(len(rules)))

# Confidence Level = 10%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
getCountList.append(len(rules))
print("Rules generated at 10% Confidence:  " + str(len(rules)))


# Question 5

frequent_itemsets = apriori(df2, min_support=0.05, use_colnames=True)

# Confidence Level = 90%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
getCountList.append(len(rules))
print("Rules generated at 90% Confidence:  " + str(len(rules)))

# Confidence Level = 80%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)
getCountList.append(len(rules))
print("Rules generated at 80% Confidence:  " + str(len(rules)))

# Confidence Level = 70%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
getCountList.append(len(rules))
print("Rules generated at 70% Confidence:  " + str(len(rules)))

# Confidence Level = 60%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
getCountList.append(len(rules))
print("Rules generated at 60% Confidence:  " + str(len(rules)))

# Confidence Level = 50%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
getCountList.append(len(rules))
print("Rules generated at 50% Confidence:  " + str(len(rules)))

# Confidence Level = 40%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
getCountList.append(len(rules))
print("Rules generated at 40% Confidence:  " + str(len(rules)))

# Confidence Level = 30%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
getCountList.append(len(rules))
print("Rules generated at 30% Confidence:  " + str(len(rules)))

# Confidence Level = 20%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)
getCountList.append(len(rules))
print("Rules generated at 20% Confidence:  " + str(len(rules)))

# Confidence Level = 10%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
getCountList.append(len(rules))
print("Rules generated at 10% Confidence:  " + str(len(rules)))


# Question 6

frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)

# Confidence Level = 90%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
getCountList.append(len(rules))

print("Rules generated at 90% Confidence:  " + str(len(rules)))

# Confidence Level = 80%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)
getCountList.append(len(rules))
print("Rules generated at 80% Confidence:  " + str(len(rules)))

# Confidence Level = 70%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
getCountList.append(len(rules))
print("Rules generated at 70% Confidence:  " + str(len(rules)))

# Confidence Level = 60%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
getCountList.append(len(rules))
print("Rules generated at 60% Confidence:  " + str(len(rules)))

# Confidence Level = 50%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
getCountList.append(len(rules))
print("Rules generated at 50% Confidence:  " + str(len(rules)))

# Confidence Level = 40%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
getCountList.append(len(rules))
print("Rules generated at 40% Confidence:  " + str(len(rules)))

# Confidence Level = 30%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
getCountList.append(len(rules))
print("Rules generated at 30% Confidence:  " + str(len(rules)))

# Confidence Level = 20%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)
getCountList.append(len(rules))
print("Rules generated at 20% Confidence:  " + str(len(rules)))

# Confidence Level = 10%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
getCountList.append(len(rules))
print("Rules generated at 10% Confidence:  " + str(len(rules)))

# Question 7

frequent_itemsets = apriori(df2, min_support=0.0005, use_colnames=True)

# Confidence Level = 90%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
getCountList.append(len(rules))

print("Rules generated at 90% Confidence:  " + str(len(rules)))

# Confidence Level = 80%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)
getCountList.append(len(rules))
print("Rules generated at 80% Confidence:  " + str(len(rules)))

# Confidence Level = 70%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
getCountList.append(len(rules))
print("Rules generated at 70% Confidence:  " + str(len(rules)))

# Confidence Level = 60%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
getCountList.append(len(rules))
print("Rules generated at 60% Confidence:  " + str(len(rules)))

# Confidence Level = 50%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
getCountList.append(len(rules))
print("Rules generated at 50% Confidence:  " + str(len(rules)))

# Confidence Level = 40%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
getCountList.append(len(rules))
print("Rules generated at 40% Confidence:  " + str(len(rules)))

# Confidence Level = 30%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
getCountList.append(len(rules))
print("Rules generated at 30% Confidence:  " + str(len(rules)))

# Confidence Level = 20%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.2)
getCountList.append(len(rules))
print("Rules generated at 20% Confidence:  " + str(len(rules)))

# Confidence Level = 10%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
getCountList.append(len(rules))
print("Rules generated at 10% Confidence:  " + str(len(rules)))

height = [10, 24, 36, 40, 5]
tick_label = ['90%', '80%', '70%', '60%', '50%', '40%', '30%', '20%', '10%']
fig = plt.figure(figsize=(10, 5))

plt.bar(tick_label, getCountList, color='blue', width=0.4)
plt.xlabel('Confidence Level')
plt.ylabel('Generated Rules')
plt.title('With Support Level 0.5%')

plt.show()
