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


def gen_cv_predictions(df, cache_results=True, non_sparse=True):
    """ Generates predictions for 10-fold cross validation
    """

    # generate the 10 fold splits
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)

    all_lin_preds = []
    all_rf_preds = []

    fold_n = 0

    # logistic regression model with defaults
    lin_cl = LinearRegression()
    # rf model
    rf_cl = RandomForestRegressor(n_estimators=100, min_samples_split=16, random_state=SEED)

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
        if cache_results:
            with open("lin_preds.pkl", "wb") as f:
                pickle.dump(all_lin_preds, f)

        # only train non-sparse models if required, as they take a long time
        if non_sparse:
            # use the most important words to train RF classifier
            # take the max absolute value from all one-v-all subclassifiers
            coef = np.abs(lin_cl.coef_).mean(0)
            important_words_ind = np.argsort(coef)[-100:]

            X_train_dense = X_train[:, important_words_ind].todense()
            X_eval_dense = X_eval[:, important_words_ind].todense()

            print("Training random forest model")
            rf_cl.fit(X_train_dense, y_train)
            rf_preds = rf_cl.predict(X_eval_dense)

           # save the predictions
            if cache_results:
                all_rf_preds.append(rf_preds)
                with open("rf_preds.pkl", "wb") as f:
                    pickle.dump(all_rf_preds, f)

        fold_n += 1


def eval_model(df, lin_preds_fn="lin_preds.pkl", rf_preds_fn="rf_preds.pkl", weights=(0.8, 0.2)):
    """ evaluates the results of the 10 fold CV
    """

    # perform k-fold validation
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)
    rms_scores_lin = np.zeros(10)
    rms_scores_rf = np.zeros(10)
    rms_scores_comb = np.zeros(10)

    with open(lin_preds_fn) as f:
        all_lin_preds = pickle.load(f)

    with open(rf_preds_fn) as f:
        all_rf_preds = pickle.load(f)

    fold_n = 0

    for train_indices, fold_eval_indices in kf:
        y_eval = np.array(df)[fold_eval_indices, 4:]

        # convert to float arrays
        y_eval = np.array(y_eval, dtype="float")

        lin_preds = all_lin_preds[fold_n]
        # probabilities for certain predictions should sum to 1
        # normalise the 'S' predictions
        print lin_preds[:, 5:9].sum(1, keepdims=True)
        lin_preds[:, 0:5] /= lin_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        lin_preds[:, 5:9] /= lin_preds[:, 5:9].sum(1, keepdims=True)

        rms_scores_lin[fold_n] = np.sqrt(np.sum(np.array(np.array(lin_preds-y_eval)**2)/(len(fold_eval_indices) * 24.0)))

        rf_preds = all_rf_preds[fold_n]
        # normalise the 'S' predictions
        rf_preds[:, 0:5] /= rf_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        rf_preds[:, 5:9] /= rf_preds[:, 5:9].sum(1, keepdims=True)
        rms_scores_rf[fold_n] = np.sqrt(np.sum(np.array(np.array(rf_preds-y_eval)**2)/(len(fold_eval_indices)*24.0)))

        #combine predictions
        comb_preds = weights[0]*lin_preds + weights[1]*rf_preds
        rms_scores_comb[fold_n] = np.sqrt(np.sum(np.array(np.array(comb_preds-y_eval)**2)/(len(fold_eval_indices)*24.0)))

        fold_n += 1

    print("Mean Linear RMS error:{}, Std:{}".format(np.mean(rms_scores_lin), np.std(rms_scores_lin)))
    print("Mean RF RMS error:{}, Std:{}".format(np.mean(rms_scores_rf), np.std(rms_scores_rf)))
    print("Mean Combined RMS error:{}, Std:{}".format(np.mean(rms_scores_comb), np.std(rms_scores_comb)))

if __name__ == "__main__":
    #df = load_raw_tweets()
    df = pd.read_csv(os.path.join("data", "train.csv"))
    #gen_cv_predictions(df)
    eval_model(df)
