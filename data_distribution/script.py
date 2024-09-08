import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/mbouharoun/ARAM/NF-UNSW-NB15.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)



train_data = pd.DataFrame(X_train)
train_labels = pd.DataFrame(y_train)
test_data = pd.DataFrame(X_test)
test_labels = pd.DataFrame(y_test)

# Concatenate data and labels
train_dataset = pd.concat([train_data, train_labels], axis=1)
test_dataset = pd.concat([test_data, test_labels], axis=1)


print(train_dataset.head())
print(test_dataset.head())

train_dataset.to_csv('/home/mbouharoun/ARAM/NF-UNSW-NB15_train.csv', index=False)

# Save the test dataset to a CSV file
test_dataset.to_csv('/home/mbouharoun/ARAM/NF-UNSW-NB15_test.csv', index=False)
