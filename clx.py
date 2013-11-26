import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

SEED = 42


def remove_http_links(concat_tweets):
    print("Removing HTTP links")
    return [" ".join([w for w in tweets.lower().split(" ") if w[0:4] != "http"])
            for tweets in concat_tweets]


def remove_stopwords(concat_tweets):
    print("Removing Stopwords")
    tokenizer = RegexpTokenizer(r'\w+')
    return [" ".join([w for w in tokenizer.tokenize(tweets.lower())
                      if w not in stopwords.words("english")])
            for tweets in concat_tweets]


def gen_cv_predictions(df, lin_preds_fn="lin_preds.pkl", ridge_preds_fn="ridge_preds.pkl", non_sparse=True):
    """ Generates predictions for 10-fold cross validation
    """

    # generate the 10 fold splits
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)

    all_lin_preds = []
    all_ridge_preds = []

    fold_n = 0

    # logistic regression model with defaults
    lin_cl = LinearRegression()
    # rf model
    ridge_cl = Ridge(alpha=3.0)

    for train_indices, fold_eval_indices in kf:
        print("Evaluating fold {} of {}".format(fold_n+1, 10))
        # take a tfidf vectorisation of the text
        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              decode_error='ignore',
                              ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)

        X_train = tfv.fit_transform(df["tweet"][train_indices])
        X_eval = tfv.transform(df["tweet"][fold_eval_indices])

        # extract all the y values, which are in column 4 onwards
        y_train = np.array(df)[train_indices, 4:]
        y_eval = np.array(df)[fold_eval_indices, 4:]

        # convert to float arrays
        y_train = np.array(y_train, dtype="float")

        print("Training linear model")
        lin_cl.fit(X_train, y_train)
        lin_preds = lin_cl.predict(X_eval)

        # save the predictions
        all_lin_preds.append(lin_preds)
        with open(lin_preds_fn, "wb") as f:
            pickle.dump(all_lin_preds, f)

        print("Training ridge regression model")
        ridge_cl.fit(X_train, y_train)
        ridge_preds = ridge_cl.predict(X_eval)

       # save the predictions
        all_ridge_preds.append(ridge_preds)
        with open(ridge_preds_fn, "wb") as f:
            pickle.dump(all_ridge_preds, f)

        fold_n += 1


def eval_model(df, lin_preds_fn="lin_preds.pkl", ridge_preds_fn="ridge_preds.pkl", weights=(0.1, 0.9)):
    """ evaluates the results of the 10 fold CV
    """

    # perform k-fold validation
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)
    rms_scores_lin = np.zeros(10)
    rms_scores_ridge = np.zeros(10)
    rms_scores_comb = np.zeros(10)

    with open(lin_preds_fn) as f:
        all_lin_preds = pickle.load(f)

    with open(ridge_preds_fn) as f:
        all_ridge_preds = pickle.load(f)

    fold_n = 0

    for train_indices, fold_eval_indices in kf:
        y_eval = np.array(df)[fold_eval_indices, 4:]

        # convert to float arrays
        y_eval = np.array(y_eval, dtype="float")

        lin_preds = all_lin_preds[fold_n]
        # probabilities for certain predictions should sum to 1
        # normalise the 'S' predictions
        lin_preds[:, 0:5] /= lin_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        lin_preds[:, 5:9] /= lin_preds[:, 5:9].sum(1, keepdims=True)

        rms_scores_lin[fold_n] = np.sqrt(np.sum(np.array(np.array(lin_preds-y_eval)**2)/(len(fold_eval_indices) * 24.0)))

        ridge_preds = all_ridge_preds[fold_n]
        # normalise the 'S' predictions
        ridge_preds[:, 0:5] /= ridge_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        ridge_preds[:, 5:9] /= ridge_preds[:, 5:9].sum(1, keepdims=True)
        rms_scores_ridge[fold_n] = np.sqrt(np.sum(np.array(np.array(ridge_preds-y_eval)**2)/(len(fold_eval_indices)*24.0)))

        #combine predictions
        comb_preds = weights[0]*lin_preds + weights[1]*ridge_preds
        rms_scores_comb[fold_n] = np.sqrt(np.sum(np.array(np.array(comb_preds-y_eval)**2)/(len(fold_eval_indices)*24.0)))

        fold_n += 1

    print("Mean Linear RMS error:{}, Std:{}".format(np.mean(rms_scores_lin), np.std(rms_scores_lin)))
    print("Mean Ridge RMS error:{}, Std:{}".format(np.mean(rms_scores_ridge), np.std(rms_scores_ridge)))
    print("Mean Combined RMS error:{}, Std:{}".format(np.mean(rms_scores_comb), np.std(rms_scores_comb)))

if __name__ == "__main__":
    #df = load_raw_tweets()
    df = pd.read_csv(os.path.join("data", "train.csv"))
    #gen_cv_predictions(df)
    eval_model(df)
