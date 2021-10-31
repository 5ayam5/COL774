import argparse
from os import makedirs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=str,
                        default='cd', help='Part number')
    parser.add_argument('-train_data', type=str,
                        default='data/poker-hand-training.data', help='Training data')
    parser.add_argument('-test_data', type=str,
                        default='data/poker-hand-testing.data', help='Test data')
    parser.add_argument('-validation_data', type=str,
                        default='data/poker-hand-validation.data', help='Validation data')
    parser.add_argument('-best', type=str, help='Best params')
    parser.add_argument('-output', type=str,
                        default='./output', help='Output directory')
    args = parser.parse_args()
    makedirs(args.output, exist_ok=True)
    return args


def extract_data(file: str):
    data = np.genfromtxt(file, delimiter=';', dtype=None, encoding=None)
    X, Y = data[1:, :-1], data[1:, -1]
    Y = np.where(Y == '\"yes\"', True, False)
    return X, Y


def one_hot_encoder(arr: np.ndarray, mapping: dict):
    tot_features = arr.shape[1]
    for mapp in mapping:
        tot_features += max(0, len(mapp) - 1)
    one_hot = [[False for _ in range(tot_features)] for _ in range(len(arr))]
    for i, example in enumerate(arr):
        idx = 0
        for j, value in enumerate(example):
            if len(mapping[j]) > 0:
                one_hot[i][idx + mapped_value(j, value, mapping)] = True
                idx += len(mapping[j])
            else:
                one_hot[i][idx] = mapped_value(j, value, mapping)
                idx += 1

    return one_hot


def map_to_int(X: np.ndarray):
    n = X.shape[1]
    mapping = [dict() for _ in range(n)]
    for example in X:
        for i, value in enumerate(example):
            try:
                int(value)
            except ValueError:
                if value not in mapping[i]:
                    mapping[i][value] = len(mapping[i])
    return mapping


def mapped_value(feature: int, val, mapping: dict):
    try:
        return int(val)
    except ValueError:
        return mapping[feature][val]


if __name__ == '__main__':
    args = parse_args()
    X_train, Y_train = extract_data(args.train_data)
    mapping = map_to_int(X_train)
    X_train = one_hot_encoder(X_train, mapping)
    X_test, Y_test = extract_data(args.test_data)
    X_test = one_hot_encoder(X_test, mapping)
    X_valid, Y_valid = extract_data(args.validation_data)
    X_valid = one_hot_encoder(X_valid, mapping)

    if args.question.find('c') != -1 or (args.question.find('d') != -1 and not args.best):
        n_estimators = [50, 150, 250, 350, 450]
        max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
        min_samples_split = [2, 4, 6, 8, 10]
        best_oob = 0
        best_params = [None, None, None]
        with open(f'{args.output}/c', 'w+') as f:
            f.write('n_estimators,max_features,min_samples_split,train_accuracy,test_accuracy,valid_accuracy,oob_accuracy\n')
            for n_estimate in n_estimators:
                for max_feature in max_features:
                    for min_sample_split in min_samples_split:
                        clf = RandomForestClassifier(n_estimators=n_estimate, max_features=max_feature,
                                                    min_samples_split=min_sample_split, oob_score=True)
                        clf.fit(X_train, Y_train)
                        train_accuracy = clf.score(X_train, Y_train)
                        test_accuracy = clf.score(X_test, Y_test)
                        valid_accuracy = clf.score(X_valid, Y_valid)
                        oob_accuracy = clf.oob_score_
                        if oob_accuracy > best_oob:
                            best_oob = oob_accuracy
                            best_params = [n_estimate, max_feature, min_sample_split]
                        print(oob_accuracy, n_estimate, max_feature, min_sample_split)
                        f.write(f'{n_estimate},{max_feature},{min_sample_split},{train_accuracy},{test_accuracy},{valid_accuracy},{oob_accuracy}\n')
        with open(f'{args.output}/c_best', 'w+') as f:
            f.write('n_estimators,max_features,min_samples_split,oob_accuracy\n')
            f.write(f'{best_params[0]},{best_params[1]},{best_params[2]},{best_oob}\n')

    if args.question.find('d') != -1 and args.best:
        with open(f'{args.output}/c_best', 'r') as f:
            f.readline()
            best_params = list(map(float, f.readline().split(',')))

    if args.question.find('d') != -1:
        n_estimators = [50, 150, 250, 350, 450]
        max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
        min_samples_split = [2, 4, 6, 8, 10]

        with open(f'{args.output}/d_n', 'w+') as f:
            f.write('n_estimators,train_accuracy,test_accuracy,valid_accuracy,oob_accuracy\n')
            for n_estimate in n_estimators:
                clf = RandomForestClassifier(n_estimators=n_estimate, max_features=best_params[1],
                                            min_samples_split=best_params[1], oob_score=True)
                clf.fit(X_train, Y_train)
                train_accuracy = clf.score(X_train, Y_train)
                test_accuracy = clf.score(X_test, Y_test)
                valid_accuracy = clf.score(X_valid, Y_valid)
                oob_accuracy = clf.oob_score_
                f.write(f'{n_estimate},{train_accuracy},{test_accuracy},{valid_accuracy},{oob_accuracy}\n')

        with open(f'{args.output}/d_m', 'w+') as f:
            f.write('max_features,train_accuracy,test_accuracy,valid_accuracy,oob_accuracy\n')
            for max_feature in max_features:
                clf = RandomForestClassifier(n_estimators=best_params[0], max_features=max_feature,
                                            min_samples_split=best_params[2], oob_score=True)
                clf.fit(X_train, Y_train)
                train_accuracy = clf.score(X_train, Y_train)
                test_accuracy = clf.score(X_test, Y_test)
                valid_accuracy = clf.score(X_valid, Y_valid)
                oob_accuracy = clf.oob_score_
                f.write(f'{max_feature},{train_accuracy},{test_accuracy},{valid_accuracy},{oob_accuracy}\n')

        with open(f'{args.output}/d_s', 'w+') as f:
            f.write('min_samples_split,train_accuracy,test_accuracy,valid_accuracy,oob_accuracy\n')
            for min_sample_split in min_samples_split:
                clf = RandomForestClassifier(n_estimators=best_params[0], max_features=best_params[1],
                                            min_samples_split=min_sample_split, oob_score=True)
                clf.fit(X_train, Y_train)
                train_accuracy = clf.score(X_train, Y_train)
                test_accuracy = clf.score(X_test, Y_test)
                valid_accuracy = clf.score(X_valid, Y_valid)
                oob_accuracy = clf.oob_score_
                f.write(f'{min_sample_split},{train_accuracy},{test_accuracy},{valid_accuracy},{oob_accuracy}\n')

        with open(f'{args.output}/d_n') as f:
            df = pd.read_csv(f)
            df.columns = ['n_estimators', 'train_accuracy', 'test_accuracy', 'valid_accuracy', 'oob_accuracy']
            df.plot(x='n_estimators', y=['train_accuracy', 'test_accuracy', 'valid_accuracy'],
                    kind='line', title='n_estimators')
            plt.savefig(f'{args.output}/d_n.png')
            plt.clf()

        with open(f'{args.output}/d_m') as f:
            df = pd.read_csv(f)
            df.columns = ['max_features', 'train_accuracy', 'test_accuracy', 'valid_accuracy', 'oob_accuracy']
            df.plot(x='max_features', y=['train_accuracy', 'test_accuracy', 'valid_accuracy'],
                    kind='line', title='max_features')
            plt.savefig(f'{args.output}/d_m.png')
            plt.clf()

        with open(f'{args.output}/d_s') as f:
            df = pd.read_csv(f)
            df.columns = ['min_samples_split', 'train_accuracy', 'test_accuracy', 'valid_accuracy', 'oob_accuracy']
            df.plot(x='min_samples_split', y=['train_accuracy', 'test_accuracy', 'valid_accuracy'],
                    kind='line', title='min_samples_split')
            plt.savefig(f'{args.output}/d_s.png')
