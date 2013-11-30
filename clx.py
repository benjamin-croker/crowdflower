import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

SEED = 42


def lower_case(tweets):
    print("Making tweets lower case")
    return [tweet.lower() for tweet in tweets]


def remove_stopwords(tweets):
    print("Removing Stopwords")
    tokenizer = RegexpTokenizer(r'\w+')
    return [" ".join([w for w in tokenizer.tokenize(tweet.lower())
                      if w not in stopwords.words("english")])
            for tweet in tweets]


def gen_cv_predictions(df, ridge_preds_fn="ridge_preds.pkl"):
    """ Generates predictions for 10-fold cross validation
    """

    # generate the 10 fold splits
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)

    all_ridge_preds = []

    fold_n = 0

    # ridge regression model
    ridge_cl = Ridge(alpha=3.0)

    # pre-process the tweets
    df["tweet"] = lower_case(df["tweet"])

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

        # convert to float arrays
        y_train = np.array(y_train, dtype="float")

        print("Training ridge regression model")
        ridge_cl.fit(X_train, y_train)
        ridge_preds = ridge_cl.predict(X_eval)

        # save the predictions
        all_ridge_preds.append(ridge_preds)
        with open(ridge_preds_fn, "wb") as f:
            pickle.dump(all_ridge_preds, f)

        fold_n += 1


def eval_model(df, lasso_preds_fn="lasso_preds.pkl", ridge_preds_fn="ridge_preds.pkl", weights=(0.05, 0.95)):
    """ evaluates the results of the 10 fold CV
    """

    # perform k-fold validation
    kf = KFold(n=df.shape[0], n_folds=10, random_state=SEED, shuffle=True)
    rms_scores_ridge = np.zeros(10)

    with open(ridge_preds_fn) as f:
        all_ridge_preds = pickle.load(f)

    fold_n = 0

    for train_indices, fold_eval_indices in kf:
        y_eval = np.array(df)[fold_eval_indices, 4:]

        # convert to float arrays
        y_eval = np.array(y_eval, dtype="float")

        ridge_preds = all_ridge_preds[fold_n]

        # no prediction should be less than zero
        ridge_preds[ridge_preds < 0] = 0

        # normalise the 'S' predictions
        ridge_preds[:, 0:5] /= ridge_preds[:, 0:5].sum(1, keepdims=True)
        # normalise the 'W' predictions
        ridge_preds[:, 5:9] /= ridge_preds[:, 5:9].sum(1, keepdims=True)
        rms_scores_ridge[fold_n] = np.sqrt(np.sum(np.array(np.array(ridge_preds-y_eval)**2)/(len(fold_eval_indices)*24.0)))

        fold_n += 1

    print("Mean Ridge RMS error:{}, Std:{}".format(np.mean(rms_scores_ridge), np.std(rms_scores_ridge)))


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

    # no prediction should be less than zero
    preds[preds < 0] = 0

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

    # gen_cv_predictions(train_DF)
    # eval_model(train_DF)

    submission(train_DF, test_DF)
    