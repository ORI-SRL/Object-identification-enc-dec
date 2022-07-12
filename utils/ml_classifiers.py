import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def svm_classifier(train_data, train_labels, test_data, test_labels, learn=False):
    # find optimum parameters for svm classification with different kernels
    # the poly kernel is slowest so use random search rather than grid search
    if learn:
        C_range = np.logspace(0, 4, 5)
        gamma_range = np.logspace(-3, 1, 5)
        deg_range = np.linspace(2, 3, 2)
        cv = StratifiedKFold(n_splits=4, shuffle=False)

        # Apply a linear kernel
        lin_param_grid = dict(C=C_range, kernel=['linear'])
        lin_grid = GridSearchCV(SVC(), param_grid=lin_param_grid, cv=cv)
        lin_grid.fit(train_data, train_labels)
        lin_params = list(lin_grid.best_params_.values())  # list([100, 'linear'])  #

        # polynomial kernel
        poly_param_grid = dict(gamma=gamma_range, C=C_range, degree=deg_range, kernel=['poly'],
                               decision_function_shape=['ovr'])
        poly_grid = RandomizedSearchCV(SVC(), param_distributions=poly_param_grid, cv=cv, n_iter=10,
                                       return_train_score=1,
                                       n_jobs=-1)
        poly_grid.fit(train_data, train_labels)  # kern: poly, gamma: 0.1, degree: 3, C: 10000
        poly_params = list(
            poly_grid.best_params_.values())  # list(['poly', 10, 2.0, 'ovr', 10]) # list([10, 'linear'])  #

        # rbf kernel
        rbf_param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'], decision_function_shape=['ovr'])
        rbf_grid = RandomizedSearchCV(SVC(), param_distributions=rbf_param_grid, cv=cv, n_iter=50, return_train_score=1)
        rbf_grid.fit(train_data, train_labels)  # 0.01, 1000
        rbf_params = list(rbf_grid.best_params_.values())  # list(['rbf', 0.001, 'ovr', 100.0])
        names = [
            "Linear SVM",
            "Poly SVM",
            "RBF SVM"]
        classifiers = [
            SVC(kernel="linear", C=lin_params[0], decision_function_shape='ovr'),
            SVC(kernel="rbf", gamma=rbf_params[1], C=rbf_params[3], decision_function_shape='ovr'),
            SVC(kernel="poly", degree=poly_params[2], gamma=poly_params[1], C=poly_params[4],
                decision_function_shape='ovr')
        ]

        most_acc = compare_classifiers(classifiers, names, train_data, train_labels, test_data, test_labels)
    else:
        svm = pickle.load(open('svm_params.py', 'rb'))
        svm.fit(train_data, train_labels)
        score = svm.score(test_data, test_labels)
        svm.predict(test_data)
        most_acc = svm

    return most_acc


def tree_searches(train_data, train_labels, test_data, test_labels, n_grasps, learn=False):
    folder = './saved_model_states/ml_states/'
    state_file = f'{folder}{n_grasps}_grasps_tree_params.pt'
    if learn:
        # similarly, here, learn the best parameters for clustering tree searches
        # Use GridSearches to find the optimum parameters for the decision trees
        depth_range = np.linspace(2, 16, 15, dtype='int')
        estimator_range = np.linspace(5, 35, 15, dtype='int')
        # feature_range = np.linspace(7, 7, 1, dtype='int')
        tree_params_grid = dict(max_depth=depth_range)
        cv = StratifiedKFold(n_splits=4, shuffle=False)

        # decision tree classifier
        tree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_params_grid, cv=cv)
        tree_grid.fit(train_data, train_labels)
        tree_params = list(tree_grid.best_params_.values())

        # random forest classifier
        forest_params_grid = dict(max_depth=depth_range, n_estimators=estimator_range)  # , max_features=feature_range)
        forest_grid = GridSearchCV(RandomForestClassifier(), param_grid=forest_params_grid, cv=cv)
        forest_grid.fit(train_data, train_labels)
        forest_params = list(forest_grid.best_params_.values())

        names = [
            "Decision Tree",
            "Random Forest"]
        classifiers = [
            DecisionTreeClassifier(max_depth=tree_params[0]),
            RandomForestClassifier(max_depth=forest_params[0],  n_estimators=forest_params[1])
                                   # max_features=forest_params[1])
        ]
        tree, score = compare_classifiers(classifiers, names, train_data, train_labels, test_data, test_labels)
    else:
        tree = pickle.load(open(state_file, 'rb'))
        tree.fit(train_data, train_labels)
        score = tree.score(test_data, test_labels)
        tree.predict(test_data)
        tree = tree
    plot_confusion(test_data, test_labels, tree, n_grasps)
    pickle.dump(tree, open(state_file, 'wb'))
    return tree, score


def knn_classifier(train_data, train_labels, test_data, test_labels, n_grasps, learn=False):
    folder = './saved_model_states/ml_states/'
    state_file = f'{folder}{n_grasps}_grasps_knn_params.pt'
    if learn:
        cv = StratifiedKFold(n_splits=4, shuffle=False)
        knn_range = list(range(1, 31))
        knn_params_grid = dict(n_neighbors=knn_range)
        knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params_grid, cv=cv)
        knn_grid.fit(train_data, train_labels)
        knn_params = list(knn_grid.best_params_.values())

        names = [
            "Nearest Neighbors",
            "Weighted KNN"]
        classifiers = [
            KNeighborsClassifier(n_neighbors=knn_params[0]),
            KNeighborsClassifier(n_neighbors=knn_params[0], weights='distance')]
        knn, score = compare_classifiers(classifiers, names, train_data, train_labels, test_data, test_labels)
    else:
        knn = pickle.load(open(state_file, 'rb'))
        knn.fit(train_data, train_labels)
        score = knn.score(test_data, test_labels)
        predict_labels = knn.predict(test_data)

        knn = knn
    plot_confusion(test_data, test_labels, knn, n_grasps)
    pickle.dump(knn, open(state_file, 'wb'))
    return knn, score


def compare_classifiers(classifiers, names, train_data, train_labels, test_data, test_labels):
    i = 0
    score_array = {}

    for name, clf in zip(names, classifiers):
        # ax = plt.subplot(1, len(classifiers) + 1, i)
        i += 1
        clf.fit(train_data, train_labels)
        score = clf.score(test_data, test_labels)
        score_array[clf] = score
        print_string = name + " " + "{:.3f}".format(score)
        print(print_string)

    most_acc = max(score_array, key=score_array.get)
    score = max(score_array.values())
    return most_acc, score


def plot_confusion(data, labels, model_fit, n_grasps):
    unique_labels = sorted(list(set(labels)))
    pred_labels = model_fit.predict(data)
    cm = confusion_matrix(labels, pred_labels, labels=unique_labels)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot()
    cm_display.ax_.set_title(f'{n_grasps} grasps - {model_fit.__class__.__name__}')
    fig = plt.figure()
    cm_display_percentages = sns.heatmap(cm / (len(labels) / len(unique_labels)),
                                         annot=True, fmt='.2%', cmap='Blues', xticklabels=unique_labels,
                                         yticklabels=unique_labels).plot()
