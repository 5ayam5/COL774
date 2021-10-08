from q1_util import *
import argparse
from os import makedirs


def get_args():
    parser = argparse.ArgumentParser(description="Q2 of COL774 Assignment")
    parser.add_argument('train', type=str, help="path of the training data")
    parser.add_argument('test', type=str, help="path of test data")
    parser.add_argument('part', type=str, help="which part number")
    parser.add_argument('-o', dest='output', type=str,
                        default='./output', help="output directory")

    args = parser.parse_args()
    makedirs(args.output, exist_ok=True)
    return args


def nb_util(training_data, test_data, output, extra=False, ignore=False):
    print("Training model...")
    vocab, n, Phi = train_model(*training_data, 5)
    print("Model trained!\nMaking predictions and writing output to file...")
    naive_bayes = np.vectorize(lambda x: predict_nb(vocab, n, Phi, x, ignore))
    m = test_data[1].shape[0]
    training_pred = naive_bayes(training_data[0])
    test_pred = naive_bayes(test_data[0])
    confusion_test = confusion_matrix(test_data[1], test_pred, 5)
    with open(output, 'w+') as f:
        f.write("train_accuracy   = {}\n".format(
                accuracy(training_pred, training_data[1])))
        f.write("test_accuracy    = {}\n".format(
                accuracy(test_pred, test_data[1])))
        f.write("test f1-score   = {}\n".format(f1_score(confusion_test)))

        if extra:
            rand_pred = np.random.randint(1, 6, m)
            f.write("random_accuracy  = {}\n".format(
                    accuracy(rand_pred, test_data[1])))
            f.write(
                "random f1-score  = {}\n".format(f1_score(confusion_matrix(test_data[1], rand_pred, 5))))

            mode_pred = np.full(m, Counter(test_data[1]).most_common(1)[0][0])
            f.write("mode_accuracy    = {}\n".format(accuracy(
                mode_pred, test_data[1])))
            f.write(
                "mode f1-score    = {}\n".format(f1_score(confusion_matrix(test_data[1], mode_pred, 5))))

        f.write("confusion_matrix (training) =\n{}\n".format(
            confusion_matrix(training_data[1], training_pred, 5)))
        f.write("confusion_matrix (test) =\n{}".format(
                confusion_test))
    print(f"Output written to {output}!")


if __name__ == "__main__":
    args = get_args()

    if args.part.find('a') != -1 or args.part.find('b') != -1 or args.part.find('c') != -1:
        training_default, test_default = gen_train_test(
            args.train, args.test, str.split)
        nb_util(training_default, test_default, args.output + "/abc", True)

    if args.part.find('d') != -1:
        training_clean, test_clean = gen_train_test(
            args.train, args.test, stem_split)
        nb_util(training_clean, test_clean, args.output + "/d")

    if args.part.find('e') != -1:
        training_bigram_clean, test__bigram_clean = gen_train_test(
            args.train, args.test, bigram_split)
        nb_util(training_bigram_clean, test__bigram_clean,
                args.output + "/e_bigram_clean")

        training_bigram, test_bigram = gen_train_test(
            args.train, args.test, bigram_split_alter)
        nb_util(training_bigram, test_bigram,
                args.output + "/e_bigram_original")

        training_trigram, test__trigram = gen_train_test(
            args.train, args.test, trigram_split)
        nb_util(training_trigram, test__trigram, args.output + "/e_trigram")

        training_ignore, test_ignore = gen_train_test(
            args.train, args.test, str.split)
        nb_util(training_ignore, test_ignore,
                args.output + "/e_ignore", ignore=True)

        training_bigram_ignore, test_bigram_ignore = gen_train_test(
            args.train, args.test, bigram_split_alter)
        nb_util(training_bigram_ignore, test_bigram_ignore,
                args.output + "/e_bigram_ignore", ignore=True)

    if args.part.find('g') != -1:
        training_summary, test_summary = gen_train_test(
            args.train, args.test, str.split, 'summary')
        training, test = gen_train_test(args.train, args.test, stem_split)
        training[0] = combine(
            training[0], training_summary[0], lambda x: 3 * x)
        test[0] = combine(test[0], test_summary[0], lambda x: 3 * x)
        nb_util(training, test, args.output + "/g")
