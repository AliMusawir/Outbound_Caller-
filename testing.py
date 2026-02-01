# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import pandas as pd
# df=pd.read_csv(r"D:\Downloads\titanic.csv")
# print(df.head())
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# df['Cabin'] = df['Cabin'].fillna('Unknown')
# df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
# df = df.drop(['Fare','Ticket','Cabin','Name','PassengerId'], axis=1)
# print(df.head())
# print("Testing script started.")
# data = load_iris()

# X = data.data   
# y = data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")



#max number in the list
def max(list):
    max_number=list[0]
    for i in list:
        if i>max_number:
            max_number=i
    return max_number
print(max([1,10,3,12,5,6,7,8,9,10]))