import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import sys

class NaiveBayes:
    def fit(self, X, y):
        n_sam, n_feat = X.shape
        self._class = np.unique(y)
        n_classes = len(self._class) 
        

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_feat), dtype=np.float64)
        self._var = np.zeros((n_classes, n_feat), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for enm, c in enumerate(self._class):
            X_c = X[y == c] 
            self._mean[enm, :] = X_c.mean(axis=0)
            self._var[enm, :] = X_c.var(axis=0)
            self._priors[enm] = X_c.shape[0] / float(n_sam)

    def pred(self, X):
        y_pred = [self._pred(x) for x in X]
        return np.array(y_pred)
    
    def _pred(self, x):
        post = []

        # calculate posterior probability for each class
        for enm, c in enumerate(self._class):
            prior = np.log(self._priors[enm])
            posterior = np.sum(np.log(self.gf(enm, x)))
            posterior = prior + posterior
            post.append(posterior)

        # return class with highest posterior probability
        return self._class[np.argmax(post)]
    
    def gf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        np.seterr(divide='ignore', invalid='ignore')
        numerator = np.exp(-((x - mean) ** 2) / (2 * v))
        denominator = np.sqrt(2 * np.pi * v)
        return numerator / denominator



if __name__ == "__main__":

    dataset_name = sys.argv[1]
    count = 0
    v=1
    accuracynaivebayes = []

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    if dataset_name == "breastcancer":

        breastcancer = pd.read_csv("./dataset/breast-cancer-wisconsin.data", names=[
                                    "column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8", "column9", "column10", "decision"])

        # breastcancer.drop(["column7"], axis=1)
        breastcancer["decision"].replace([2, 4], [-1, 1], inplace=True)

        breastcancer = breastcancer.sample(frac=1)

        breastcancer = breastcancer.replace("?", np.NaN)
        breastcancer = breastcancer.dropna()

        breastcancer = breastcancer.drop("column1", axis="columns")

        breastcancer['column2'] = breastcancer['column2'].astype(int)
        breastcancer['column3'] = breastcancer['column3'].astype(int)
        breastcancer['column4'] = breastcancer['column4'].astype(int)
        breastcancer['column5'] = breastcancer['column5'].astype(int)
        breastcancer['column6'] = breastcancer['column6'].astype(int)
        breastcancer['column7'] = breastcancer['column7'].astype(int)
        breastcancer['column8'] = breastcancer['column8'].astype(int)
        breastcancer['column9'] = breastcancer['column9'].astype(int)
        breastcancer['column10'] = breastcancer['column10'].astype(int)
        breastcancer['decision'] = breastcancer['decision'].astype(int)

        nb = NaiveBayes()
        
        for i in range(10):
            breastcancer = breastcancer.sample(frac=1)
            X = breastcancer.drop(["decision"], axis=1).to_numpy()
            y = breastcancer.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                nb.fit(X_train, y_train)
                predictions = nb.pred(X_test)
                
                accuracy_data = accuracy(predictions, y_test)

            accuracynaivebayes.append(accuracy_data)

            print('Accuracy for breastcancer Dataset:', accuracy_data)

        mean_accuracy = np.sum(accuracynaivebayes)/float(len(accuracynaivebayes))
        std_dev = np.std(accuracynaivebayes)

        print("Standard Deviation for breastcancer Dataset:", std_dev, " \n ")

    elif dataset_name == "car":


        car = pd.read_csv("./dataset/car.data", names=[
                                  "buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])

        # Shuffle the dataset
        car = car.sample(frac=1)

        car["decision"].replace(
            ["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        car["safety"].replace(
            ["low", "med", "high"], [0, 1, 2], inplace=True)
        car["lug_boot"].replace(
            ["small", "med", "big"], [0, 1, 2], inplace=True)
        car["persons"].replace(["more"], [4], inplace=True)
        car["doors"].replace(["5more"], [6], inplace=True)
        car["maint"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)
        car["buying"].replace(["vhigh", "high", "med", "low"], [
            4, 3, 2, 1], inplace=True)

        car['decision'] = car['decision'].astype(int)
        car['safety'] = car['safety'].astype(int)
        car['lug_boot'] = car['lug_boot'].astype(int)
        car['persons'] = car['persons'].astype(int)
        car['doors'] = car['doors'].astype(int)
        car['maint'] = car['maint'].astype(int)
        car['buying'] = car['buying'].astype(int)

        nb = NaiveBayes()
        
        for i in range(10):
            car = car.sample(frac=1)
            X = car.drop(["decision"], axis=1).to_numpy()
            y = car.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                nb.fit(X_train, y_train)
                predictions = nb.pred(X_test)
                
                accuracy_data = accuracy(predictions, y_test)

            accuracynaivebayes.append(accuracy_data)

            print('Accuracy for Car Dataset:', accuracy_data)

        mean_accuracy = np.sum(accuracynaivebayes)/float(len(accuracynaivebayes))
        std_dev = np.std(accuracynaivebayes)

        print("Standard Deviation for Car Dataset:", std_dev, " \n ")

    elif dataset_name == "mushroom":

        mushroom = pd.read_csv("./dataset/mushroom.data", names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment",
                                                                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                                                                 "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
                                                                 "population", "habitat"])

        mushroom["decision"].replace(["e", "p"], [0, 1], inplace=True)
        mushroom["cap-shape"].replace(["b", "c", "x", "f",
                                      "k", "s"], [0, 1, 2, 3, 4, 5], inplace=True)
        mushroom["cap-surface"].replace(["f", "g",
                                        "y", "s"], [0, 1, 2, 3], inplace=True)
        mushroom["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [
                                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["bruises"].replace(["t", "f"], [0, 1], inplace=True)
        mushroom["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [
                                 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["gill-attachment"].replace(["a",
                                            "d", "f", "n"], [0, 1, 2, 3], inplace=True)
        mushroom["gill-spacing"].replace(["c",
                                         "w", "d"], [0, 1, 2], inplace=True)
        mushroom["gill-size"].replace(["b", "n"], [0, 1], inplace=True)
        mushroom["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"], [
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        mushroom["stalk-shape"].replace(["e", "t"], [0, 1], inplace=True)
        mushroom["stalk-root"].replace(["b", "c", "u", "e",
                                       "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace=True)
        mushroom["stalk-surface-above-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom["stalk-surface-below-ring"].replace(
            ["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        mushroom["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
                                                   1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [
                                                   1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["veil-type"].replace(["p", "u"], [1, 2], inplace=True)
        mushroom["veil-color"].replace(["n", "o",
                                       "w", "y"], [1, 2, 3, 4], inplace=True)
        mushroom["ring-number"].replace(["n", "o", "t"],
                                        [1, 2, 3], inplace=True)
        mushroom["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [
                                      1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
        mushroom["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [
                                              1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        mushroom["population"].replace(["a", "c", "n", "s", "v", "y"], [
                                       1, 2, 3, 4, 5, 6], inplace=True)
        mushroom["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [
                                    1, 2, 3, 4, 5, 6, 7], inplace=True)

        # Shuffle the dataset
        mushroom = mushroom.sample(frac=1)

        mushroom['decision'] = mushroom['decision'].astype(int)
        mushroom['cap-shape'] = mushroom['cap-shape'].astype(int)
        mushroom['cap-surface'] = mushroom['cap-surface'].astype(int)
        mushroom['cap-color'] = mushroom['cap-color'].astype(int)
        mushroom['bruises'] = mushroom['bruises'].astype(int)
        mushroom['odor'] = mushroom['odor'].astype(int)
        mushroom['gill-attachment'] = mushroom['gill-attachment'].astype(int)
        mushroom['gill-spacing'] = mushroom['gill-spacing'].astype(int)
        mushroom['gill-size'] = mushroom['gill-size'].astype(int)
        mushroom['gill-color'] = mushroom['gill-color'].astype(int)
        mushroom['stalk-shape'] = mushroom['stalk-shape'].astype(int)
        mushroom['stalk-root'] = mushroom['stalk-root'].astype(int)
        mushroom['stalk-surface-above-ring'] = mushroom['stalk-surface-above-ring'].astype(
            int)
        mushroom['stalk-surface-below-ring'] = mushroom['stalk-surface-below-ring'].astype(
            int)
        mushroom['stalk-color-above-ring'] = mushroom['stalk-color-above-ring'].astype(
            int)
        mushroom['stalk-color-below-ring'] = mushroom['stalk-color-below-ring'].astype(
            int)
        mushroom['veil-type'] = mushroom['veil-type'].astype(int)
        mushroom['veil-color'] = mushroom['veil-color'].astype(int)
        mushroom['ring-number'] = mushroom['ring-number'].astype(int)
        mushroom['ring-type'] = mushroom['ring-type'].astype(int)
        mushroom['spore-print-color'] = mushroom['spore-print-color'].astype(
            int)
        mushroom['population'] = mushroom['population'].astype(int)
        mushroom['habitat'] = mushroom['habitat'].astype(int)

        nb = NaiveBayes()
        for i in range(10):
            mushroom = mushroom.sample(frac=1)
            X = mushroom.drop(["decision"], axis=1).to_numpy()
            y = mushroom.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                nb.fit(X_train, y_train)
                predictions = nb.pred(X_test)
                
                accuracy_data = accuracy(predictions, y_test)

            accuracynaivebayes.append(accuracy_data)

            print('Accuracy for mushroom Dataset:', accuracy_data)

        mean_accuracy = np.sum(accuracynaivebayes)/float(len(accuracynaivebayes))
        std_dev = np.std(accuracynaivebayes)

        print("Standard Deviation for mushroom Dataset:", std_dev, " \n ")

    elif dataset_name == "ecoli":

        ecoli = pd.read_csv("./dataset/ecoli.data", names=["column1", "column2", "column3", "column4",
                            "column5", "column6", "column7", "column8", "decision"], delim_whitespace=True)

        # Shuffle the dataset
        ecoli = ecoli.sample(frac=1)

        ecoli = ecoli.drop("column1", axis="columns")
        ecoli["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"], [
                                  0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        ecoli['column2'] = ecoli['column2'].astype(float)
        ecoli['column3'] = ecoli['column3'].astype(float)
        ecoli['column4'] = ecoli['column4'].astype(float)
        ecoli['column5'] = ecoli['column5'].astype(float)
        ecoli['column6'] = ecoli['column6'].astype(float)
        ecoli['column7'] = ecoli['column7'].astype(float)
        ecoli['column8'] = ecoli['column8'].astype(float)
        ecoli['decision'] = ecoli['decision'].astype(int) 
        
        nb = NaiveBayes()
        for i in range(10):
            ecoli = ecoli.sample(frac=1)
            X = ecoli.drop(["decision"], axis=1).to_numpy()
            y = ecoli.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                nb.fit(X_train, y_train)
                predictions = nb.pred(X_test)
                
                accuracy_data = accuracy(predictions, y_test)

            accuracynaivebayes.append(accuracy_data)

            print('Accuracy for ecoli Dataset:', accuracy_data)

        mean_accuracy = np.sum(accuracynaivebayes)/float(len(accuracynaivebayes))
        std_dev = np.std(accuracynaivebayes)

        print("Standard Deviation for ecoli Dataset:", std_dev, " \n ")

    elif dataset_name == "letterrecognition":

        letterrecognition = pd.read_csv("./dataset/letter-recognition.data", names=["decision", "column2", "column3", "column4", "column5", "column6",
                                        "column7", "column8", "column9", "column10", "column11", "column12", "column13", "column14", "column15", "column16", "column17"])

        # Shuffle the dataset
        letterrecognition = letterrecognition.sample(frac=1)

        letterrecognition["decision"].replace(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"], [
                                              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], inplace=True)

        letterrecognition['decision'] = letterrecognition['decision'].astype(
            int)
        letterrecognition['column2'] = letterrecognition['column2'].astype(int)
        letterrecognition['column3'] = letterrecognition['column3'].astype(int)
        letterrecognition['column4'] = letterrecognition['column4'].astype(int)
        letterrecognition['column5'] = letterrecognition['column5'].astype(int)
        letterrecognition['column6'] = letterrecognition['column6'].astype(int)
        letterrecognition['column7'] = letterrecognition['column7'].astype(int)
        letterrecognition['column8'] = letterrecognition['column8'].astype(int)
        letterrecognition['column9'] = letterrecognition['column9'].astype(int)
        letterrecognition['column10'] = letterrecognition['column10'].astype(
            int)
        letterrecognition['column11'] = letterrecognition['column7'].astype(
            int)
        letterrecognition['column12'] = letterrecognition['column8'].astype(
            int)
        letterrecognition['column13'] = letterrecognition['column9'].astype(
            int)
        letterrecognition['column14'] = letterrecognition['column10'].astype(
            int)
        letterrecognition['column15'] = letterrecognition['column8'].astype(
            int)
        letterrecognition['column16'] = letterrecognition['column9'].astype(
            int)
        letterrecognition['column17'] = letterrecognition['column10'].astype(
            int)

        nb = NaiveBayes()
        for i in range(10):
            letterrecognition = letterrecognition.sample(frac=1)
            X = letterrecognition.drop(["decision"], axis=1).to_numpy()
            y = letterrecognition.decision.to_numpy()

            kfold_knn = KFold(n_splits=5)

            for train_index, test_index in kfold_knn.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                nb.fit(X_train, y_train)
                predictions = nb.pred(X_test)
                
                accuracy_data = accuracy(predictions, y_test)

            accuracynaivebayes.append(accuracy_data)

            print('Accuracy for letterrecognition Dataset:', accuracy_data)

        mean_accuracy = np.sum(accuracynaivebayes)/float(len(accuracynaivebayes))
        std_dev = np.std(accuracynaivebayes)

        print("Standard Deviation for letterrecognition Dataset:", std_dev, " \n ")

    else:
        print("Give proper dataset")
