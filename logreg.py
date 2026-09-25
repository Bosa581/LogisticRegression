#Orobosa Igbinovia
import random
import numpy as np
from math import exp, log
from collections import defaultdict

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df, num_documents=None):
        """
        Create a new example
        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        :param df: Document frequencies from the vocabulary file
        :param num_documents: Number of documents used for those frequencies
        """
        self.nonzero = {vocab.index(kBIAS): 1}
        self.y = label
        self.x = np.zeros(len(vocab))
        for entry in words:
            word, count = entry.split(":")
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1

        self.x_tfidf = np.zeros(len(vocab))
        self.x_tfidf[0] = 1
        if df is not None and num_documents is not None:
            squared_length = 0.0
            for x in range(1, len(vocab)):
                if self.x[x] > 0:
                    tf = 1 + log(self.x[x])
                    idf = 1 + log((1 + num_documents) / (1 + df[x]))
                    self.x_tfidf[x] = tf * idf
                    squared_length += self.x_tfidf[x] ** 2

            # Normalize word features, keeping the bias separate.
            if squared_length > 0:
                length = np.sqrt(squared_length)
                for x in range(1, len(vocab)):
                    self.x_tfidf[x] = self.x_tfidf[x] / length


class LogReg:
    def __init__(self, num_features, mu, step):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter (for extra credit)
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.dimension = num_features
        self.beta = np.zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = np.zeros(num_features)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples, use_tfidf=False):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :param use_tfidf: Use the same TF-IDF features used during training
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            features = ii.x
            if use_tfidf:
                features = ii.x_tfidf
            p = sigmoid(self.beta.dot(features))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            if self.mu > 0:
                logprob -= self.mu * np.sum(self.beta ** 2)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration,
                  lazy=False, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        features = train_example.x
        if use_tfidf:
            features = train_example.x_tfidf

        if lazy:
            for x in range(len(self.beta)):
                if features[x] != 0:
                    for i in range(int(self.last_update[x] + 1), iteration):
                        self.beta[x] *= 1 - 2 * self.step(i) * self.mu

        z = 0.0
        for x in range(len(features)):
            z = z + (features[x] * self.beta[x])
        
        sigfunc  = sigmoid(z)
        if lazy == True:
            for x in range(len(self.beta)):
                if features[x] == 0:
                        continue
                else:
                    old_beta = self.beta[x]
                    self.beta[x] = old_beta + self.step(iteration)*((train_example.y - sigfunc)*features[x] - 2 * self.mu * old_beta)
                    self.last_update[x] = iteration
        else:
            for x in range(len(features)):
                old_beta = self.beta[x]
                self.beta[x] = old_beta + self.step(iteration) * (
                    (train_example.y - sigfunc) * features[x])
        

        return self.beta

    def finalize_lazy(self, iteration):
        """
        After going through all normal updates, apply regularization to
        all variables that need it.
        Only implement this function if you do the extra credit.
        """
        for x in range(len(self.beta)):
            for i in range(int(self.last_update[x]) + 1, iteration + 1):
                self.beta[x] *= 1 - 2 * self.step(i) * self.mu
            self.last_update[x] = iteration

        return self.beta

def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary
    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = []
    vocab_words = []
    with open(vocab, 'r') as vocab_file:
        for line in vocab_file:
            if '\t' in line:
                fields = line.split('\t')
                vocab_words.append(fields[0])
                df.append(float(fields[1]))
    vocab = vocab_words
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    num_documents = 0
    for filename in [positive, negative]:
        with open(filename, 'r') as input_file:
            for line in input_file:
                num_documents += 1

    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df, num_documents)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="data/positive.txt", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="data/negative.txt", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="data/vocab.txt", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--ec", help="Extra credit option (df, lazy, or rate)",
                           type=str, default="")

    args = argparser.parse_args()
    use_tfidf = args.ec == "df"
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    if args.ec != "rate":
        lr = LogReg(len(vocab), args.mu, lambda x: args.step)
    else:
        # Modify this code if you do learning rate extra credit
        lr = LogReg(len(vocab), args.mu,
            lambda t: args.step / np.sqrt(t))

    # Iterations
    update_number = 0
    for pp in range(args.passes):
        for ii in train:
            update_number += 1
            # Do we use extra credit option
            if args.ec == "df":
                lr.sg_update(ii, update_number, use_tfidf=True)
            elif args.ec == "lazy":
                lr.sg_update(ii, update_number, lazy=True)
            else:
                lr.sg_update(ii, update_number)

            if update_number % 5 == 1:
                if args.ec == "lazy":
                    lr.finalize_lazy(update_number)
                train_lp, train_acc = lr.progress(train, use_tfidf=use_tfidf)
                ho_lp, ho_acc = lr.progress(test, use_tfidf=use_tfidf)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))

    # Final update with empty example
    if args.ec == "lazy":
        lr.finalize_lazy(update_number)

    train_lp, train_acc = lr.progress(train, use_tfidf=use_tfidf)
    ho_lp, ho_acc = lr.progress(test, use_tfidf=use_tfidf)
    print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
          (update_number, train_lp, ho_lp, train_acc, ho_acc))
