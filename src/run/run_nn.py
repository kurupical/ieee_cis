import pandas as pd
import lightgbm as lgb
import os
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import gc
import time
from src.feature.common import reduce_mem_usage
from sklearn.preprocessing import StandardScaler

import keras
import random
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score

# hyper parameters
n_folds = 5
random_state = 0
n_loop = 1
target_col = "isFraud"
id_col = "TransactionID"
remove_cols = ["TransactionDT",
               "id_13", "id_30", "id_31"# train/testで分布が違いすぎる (別手法で対応),
               # "card1", "card2", "card3", "card5",
               # "addr1", "addr2"
               ]
# remove_cols.extend(pd.read_csv("cols.csv")["column"].values)

is_reduce_memory = False
# select_cols = None # 全てのcolumnを選ぶ
select_cols = pd.read_csv("cols_nn.csv")["column"].values

def _get_categorical_features(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    feats = [col for col in list(df.columns) if df[col].dtype not in numerics]

    cat_cols = []
    cat_cols.extend(["ProductCD"])
    cat_cols.extend(["card{}".format(x) for x in np.arange(1, 6+1)])
    cat_cols.extend(["addr1", "addr2"])
    cat_cols.extend(["P_emaildomain", "R_emaildomain"])
    cat_cols.extend(["M{}".format(x) for x in np.arange(1, 9+1)])
    cat_cols.extend(["DeviceType", "DeviceInfo"])
    cat_cols.extend(["id_{}".format(x) for x in np.arange(12, 38+1)])

    cat_cols = [x for x in df.columns if x in cat_cols]
    feats.extend([x for x in cat_cols if x not in feats])
    return feats

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x_train)
        y_pred_val = self.model.predict(self.x_val)
        roc_train = roc_auc_score(self.y_train, y_pred_train)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print("AUC train: {:.4f}, test: {:.4f}".format(roc_train, roc_val))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def get_model(model_name, input_shape):
    get_custom_objects().update({'custom_gelu': custom_gelu})
    get_custom_objects().update({'focal_loss_fn': focal_loss()})

    if model_name == "basic":
        inputs = Input(shape=input_shape)
        x = Dense(512, activation=custom_gelu)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Nadam(),
            loss="focal_loss_fn"
        )
        model.summary()
        return model

    if model_name == "basic_deep":
        inputs = Input(shape=input_shape)
        x = Dense(1024, activation=custom_gelu)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Nadam(),
            loss="focal_loss_fn"
        )
        model.summary()
        return model

    if model_name == "basic_deep2":
        inputs = Input(shape=input_shape)
        x = Dense(1024, activation=custom_gelu)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Nadam(),
            loss="focal_loss_fn"
        )
        model.summary()
        return model

    if model_name == "basic_deep2_ce":
        inputs = Input(shape=input_shape)
        x = Dense(1024, activation=custom_gelu)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation=custom_gelu)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Nadam(),
            loss="binary_crossentropy"
        )
        model.summary()
        return model


def learning(df_train, df_test, model_name, output_dir):

    i = 0
    folds = KFold(n_splits=n_folds)

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")

    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    print(df_test)
    df_pred_test[id_col] = df_test[id_col]
    df_result = pd.DataFrame()
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train)):
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")

        X_train = df_train.drop([id_col, target_col], axis=1).iloc[train_idx].replace(np.inf, 0).replace(-np.inf, 0).fillna(0).values.astype(np.float32)
        y_train = df_train[target_col].iloc[train_idx].values
        X_val = df_train.drop([id_col, target_col], axis=1).iloc[val_idx].replace(np.inf, 0).replace(-np.inf, 0).fillna(0).values.astype(np.float32)
        y_val = df_train[target_col].iloc[val_idx].values
        X_test = df_test.drop([id_col], axis=1).replace(np.inf, 0).replace(-np.inf, 0).fillna(0).values.astype(np.float32)
        sc = StandardScaler().fit(np.concatenate([X_train, X_val, X_test]))
        X_train = sc.transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)
        model = get_model(model_name=model_name, input_shape=(len(X_train[0]), ))
        model.fit(X_train, y_train,
                  epochs=20, batch_size=2048,
                  validation_data=(X_val, y_val),
                  verbose=True,
                  callbacks=[roc_callback(training_data=(X_train, y_train), validation_data=(X_val, y_val)),
                             ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
                             EarlyStopping(monitor="val_loss", patience=6, verbose=1),
                             ModelCheckpoint("{}/best.hdf5".format(output_dir), monitor="val_loss", verbose=1,
                                             save_best_only=True)]
                )

        model = load_model("{}/best.hdf5".format(output_dir),
                           custom_objects={"focal_loss": focal_loss,
                                           "custom_gelu": custom_gelu})
        w_pred_train = model.predict(X_val).reshape(-1)
        print(w_pred_train)
        df_pred_train = df_pred_train.append(pd.DataFrame(
            {id_col: df_train[id_col].iloc[val_idx],
             "pred": w_pred_train,
             "y": df_train[target_col].iloc[val_idx]}
        ), ignore_index=True)

        w_pred_test = model.predict(X_test).reshape(-1)
        print(w_pred_test.mean())
        df_pred_test["pred_fold{}_{}".format(i, n_fold)] = w_pred_test
        df_result = df_result.append(
            pd.DataFrame(
                {"fold": [n_fold],
                 "random_state": [random_state],
                 "auc_train": [roc_auc_score(y_train, model.predict(X_train))],
                 "auc_test": [roc_auc_score(y_val, w_pred_train)]}
            ),
            ignore_index=True
        )
        del w_pred_train, w_pred_test, model
        gc.collect()

    df_submit = pd.DataFrame()
    df_submit[id_col] = df_test[id_col]
    df_submit[target_col] = df_pred_test.drop(id_col, axis=1).mean(axis=1)
    return df_submit, df_pred_train, df_pred_test, df_result

def main(model_name="basic"):
    # print("waiting...")
    # time.sleep(60*60*0.5)
    output_dir = "../../output/{}".format(dt.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(output_dir)

    df_submit = pd.DataFrame()
    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    df_importance = pd.DataFrame()

    print("load train dataset")
    df_train = pd.read_feather("../../data/merge/train_merge.feather").drop(remove_cols, axis=1, errors="ignore")
    print("load test dataset")
    df_test = pd.read_feather("../../data/merge/test_merge.feather").drop(remove_cols, axis=1, errors="ignore")

    cat_feats = _get_categorical_features(df_train)
    df_train = df_train.drop(cat_feats, axis=1)
    df_test = df_test.drop(cat_feats, axis=1)
    if select_cols is not None:
        df_train = df_train[[x for x in (list(select_cols) + [target_col] + [id_col]) if x in df_train.columns]]
        df_test = df_test[[x for x in (list(select_cols) + [id_col]) if x in df_test.columns]]

    if is_reduce_memory:
        df_train = reduce_mem_usage(df_train)
        df_test = reduce_mem_usage(df_test)

    sub, pred_train, pred_test, result = learning(df_train=df_train,
                                                  df_test=df_test,
                                                  model_name=model_name,
                                                  output_dir=output_dir)

    df_submit = df_submit.append(sub, ignore_index=True)
    df_pred_train = df_pred_train.append(pred_train, ignore_index=True)
    df_pred_test = df_pred_test.append(pred_test, ignore_index=True)

    df_submit.to_csv("{}/submit.csv".format(output_dir), index=False)
    df_pred_train.to_csv("{}/predict_train.csv".format(output_dir), index=False)
    df_pred_test.to_csv("{}/submit_detail.csv".format(output_dir), index=False)
    result.to_csv("{}/result.csv".format(output_dir), index=False)

if __name__ == "__main__":
    main("basic_deep2")