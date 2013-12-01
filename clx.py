import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.lancaster import LancasterStemmer

SEED = 42


def lower_case(tweets):
    print("Making tweets lower case")
    return [tweet.lower() for tweet in tweets]


def remove_stopwords(tweets):
    print("Removing stopwords")
    tokenizer = RegexpTokenizer(r'\w+')

    return [" ".join([w for w in tokenizer.tokenize(tweet.lower())
                      if w not in stopwords.words("english")])
            for tweet in tweets]


def stem_words(tweets):
    print("Stemming words")
    stemmer = LancasterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')

    return [" ".join([stemmer.stem(w) for w in tokenizer.tokenize(tweet.lower())])
            for tweet in tweets]


def gen_cv_predictions(df, ridge_preds_fn="ridge_preds.pkl", state_preds_fn="state_preds.pkl"):
    """ Generates predictions for 10-fold cross validation
    """

    # generate the 10 fold splits
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)

    all_ridge_preds = []
    all_state_preds = []

    fold_n = 0

    # ridge regression model
    ridge_cl = Ridge(alpha=3.0)
    state_cl = Ridge()

    for train_indices, fold_eval_indices in kf:
        print("Evaluating fold {} of {}".format(fold_n + 1, 10))
        # take a tfidf vectorisation of the text
        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              decode_error='ignore',
                              ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)
        one_hot_enc = OneHotEncoder()
        label_enc = LabelEncoder()

        X_train_ridge = tfv.fit_transform(df["tweet"][train_indices])
        X_eval_ridge = tfv.transform(df["tweet"][fold_eval_indices])

        X_train_states = one_hot_enc.fit_transform(
            label_enc.fit_transform(df["state"])[train_indices][np.newaxis].T)
        X_eval_states = one_hot_enc.fit_transform(
            label_enc.fit_transform(df["state"])[fold_eval_indices][np.newaxis].T)

        # extract all the y values, which are in column 4 onwards
        y_train = np.array(df)[train_indices, 4:]

        # convert to float arrays
        y_train = np.array(y_train, dtype="float")

        print("Training ridge regression model")
        ridge_cl.fit(X_train_ridge, y_train)
        ridge_preds = ridge_cl.predict(X_eval_ridge)

        print("Training ridge model using states")
        state_cl.fit(X_train_states, y_train)
        state_preds = state_cl.predict(X_eval_states)

        # save the predictions
        all_ridge_preds.append(ridge_preds)
        with open(ridge_preds_fn, "wb") as f:
            pickle.dump(all_ridge_preds, f)

        all_state_preds.append(state_preds)
        with open(state_preds_fn, "wb") as f:
            pickle.dump(all_state_preds, f)

        fold_n += 1


def eval_model(df,
               ridge_preds_fn="ridge_preds.pkl",
               state_preds_fn="state_preds.pkl",
               weights=(0.9, 0.1)):
    """ evaluates the results of the 10 fold CV
    """

    # perform k-fold validation
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)
    rms_scores_ridge = np.zeros(10)
    rms_scores_state = np.zeros(10)
    rms_scores_comb = np.zeros(10)

    with open(ridge_preds_fn) as f:
        all_ridge_preds = pickle.load(f)
    with open(state_preds_fn) as f:
        all_state_preds = pickle.load(f)

    fold_n = 0

    for train_indices, fold_eval_indices in kf:
        y_eval = np.array(df)[fold_eval_indices, 4:]

        # convert to float arrays
        y_eval = np.array(y_eval, dtype="float")

        ridge_preds = all_ridge_preds[fold_n]
        # predictions tend to gravitate to 0 or 1
        ridge_preds[ridge_preds < 0.05] = 0.0
        ridge_preds[ridge_preds > 0.95] = 1.0

        # normalise the 'S' predictions
        ridge_preds[:, 0:5] /= ridge_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        ridge_preds[:, 5:9] /= ridge_preds[:, 5:9].sum(1, keepdims=True)
        rms_scores_ridge[fold_n] = np.sqrt(np.sum(np.array(np.array(ridge_preds - y_eval) ** 2) /
                                                  (len(fold_eval_indices) * 24.0)))

        state_preds = all_state_preds[fold_n]
        rms_scores_state[fold_n] = np.sqrt(np.sum(np.array(np.array(state_preds - y_eval) ** 2) /
                                                  (len(fold_eval_indices) * 24.0)))

        combined_preds = weights[0] * ridge_preds + weights[1] * state_preds
        rms_scores_comb[fold_n] = np.sqrt(np.sum(np.array(np.array(combined_preds - y_eval) ** 2) /
                                                 (len(fold_eval_indices) * 24.0)))

        fold_n += 1

    print("Mean Ridge RMS error:{}, Std:{}".format(np.mean(rms_scores_ridge), np.std(rms_scores_ridge)))
    print("Mean State RMS error:{}, Std:{}".format(np.mean(rms_scores_state), np.std(rms_scores_state)))
    print("Mean Combined RMS error:{}, Std:{}".format(np.mean(rms_scores_comb), np.std(rms_scores_comb)))


def submission(train_DF, test_DF, output_filename="submission.csv"):
    ridge_cl = Ridge(alpha=3.0)

    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}',
                          decode_error='ignore',
                          ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                          sublinear_tf=1)

    X_train = tfv.fit_transform(train_DF["tweet"])
    X_test = tfv.transform(test_DF["tweet"])

    # extract all the y values, which are in column 4 onwards
    y_train = np.array(train_DF)[:, 4:]

    # convert to float arrays
    y_train = np.array(y_train, dtype="float")

    print("Training ridge regression model")
    ridge_cl.fit(X_train, y_train)
    preds = ridge_cl.predict(X_test)

    # predictions tend to gravitate to 0 or 1
    preds[preds < 0.05] = 0.0
    preds[preds > 0.95] = 1.0

    # normalise the 'S' predictions
    preds[:, 0:5] /= preds[:, 0:5].sum(1, keepdims=True)
    # normalise the 'W' predictions
    preds[:, 5:9] /= preds[:, 5:9].sum(1, keepdims=True)

    # make a submission
    submission_DF = pd.DataFrame(data=preds, columns=train_DF.columns[4:])
    # add the id column
    submission_DF.insert(0, "id", test_DF["id"])
    submission_DF.to_csv(output_filename, index=False)


if __name__ == "__main__":
    #df = load_raw_tweets()
    train_DF = pd.read_csv(os.path.join("data", "train.csv"))
    test_DF = pd.read_csv(os.path.join("data", "test.csv"))

    #gen_cv_predictions(train_DF)
    eval_model(train_DF, weights=(0.95, 0.05))

    #submission(train_DF, test_DF)
