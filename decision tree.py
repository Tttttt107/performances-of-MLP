from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier         # decision trees
from sklearn.tree import DecisionTreeRegressor          # decision trees
from sklearn import tree

data = pd.read_csv('data/processed/data.csv')
#predict the label using the features
#Split the training and test datasets

train_data, test_data = train_test_split(data, test_size=0.2)

#split the features and category in train and test data (where label is in the last column of dataset)
x_train = train_data.to_numpy()[:, :-1]
y_train = train_data.to_numpy()[:, -1]

x_test = test_data.to_numpy()[:, :-1]
y_test = test_data.to_numpy()[:, -1]

# build the model
max_depth_list = [4]

for max_depth in max_depth_list:
    dt_model = DecisionTreeClassifier(max_depth=max_depth)
    dt_model.fit(x_train, y_train)

    tree.plot_tree(dt_model)
    plt.show()

    #calculate the train accuracy and test accuracy
    train_acc = dt_model.score(x_train, y_train)
    test_acc = dt_model.score(x_test, y_test)

    print('max_depth', max_depth)
    print('Training accuracy：{:.2f}%'.format(train_acc * 100))
    print('Testing accuracy：{:.2f}%'.format(test_acc * 100))