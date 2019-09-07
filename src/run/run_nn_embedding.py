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
from keras.layers import Dense, Input, Dropout, BatchNormalization, Embedding, Flatten, concatenate, PReLU, Multiply
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

# ===============
# NN
# ===============
class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

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

def se_block(input, channels, r=8):
    x = Dense(channels//r, activation="relu")(input)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])

def get_keras_data(df, cat_feats):

    numerical_feats = [x for x in df.columns if x not in cat_feats]

    X = {
        "numerical": df[numerical_feats]
    }
    for c in cat_feats:
        X[c.replace("+", "")] = df[c]

    return X

def get_model(model_name, df, cat_feats, emb_n=8, dout=0.25):
    get_custom_objects().update({'custom_gelu': custom_gelu})
    get_custom_objects().update({'focal_loss_fn': focal_loss()})

    if model_name == "basic_wodori":
        numerical_feats = [x for x in df.columns if x not in cat_feats]
        inp_cats = []
        embs = []
        for c in cat_feats:
            inp_cat = Input(shape=[1], name=c.replace("+", ""))
            inp_cats.append(inp_cat)
            embs.append((Embedding(df[c].max() + 1, emb_n)(inp_cat)))

        cats = Flatten()(concatenate(embs))
        cats = Dense(1024, activation="custom_gelu")(cats)
        cats = BatchNormalization()(cats)
        cats = Dropout(dout)(cats)
        cats = Dense(128, activation="custom_gelu")(cats)
        cats = BatchNormalization()(cats)
        cats = Dropout(dout/2)(cats)

        inp_numerical = Input(shape=(len(numerical_feats),), name="numerical")
        nums = Dense(256, activation="custom_gelu")(inp_numerical)
        nums = BatchNormalization()(nums)
        nums = Dropout(dout)(nums)
        nums = Dense(128, activation="custom_gelu")(nums)
        nums = BatchNormalization()(nums)
        nums = Dropout(dout/2)(nums)

        x = concatenate([nums, cats])
        x = se_block(x, 128+128)
        x = BatchNormalization()(x)
        x = Dropout(dout/4)(x)

        out = Dense(1, activation="sigmoid", name="out1")(x)

        model = Model(inputs=inp_cats + [inp_numerical],
                      outputs=out)

        model.compile(optimizer=Nadam(), loss="focal_loss_fn")
        model.summary()
        return model

def learning(df_train, df_test, model_name, output_dir):

    i = 0
    folds = KFold(n_splits=n_folds)

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")
    cat_feats = ["card1", "card2", "card3", "card5", "card6", "addr1"]

    drop_cat_cols = [x for x in _get_categorical_features(df_test) if x not in cat_feats]
    df_train = df_train.drop(drop_cat_cols, axis=1)
    df_test = df_test.drop(drop_cat_cols, axis=1)

    for col in cat_feats:
        valid = pd.concat([df_train[[col]], df_test[[col]]])
        valid = valid[col].value_counts()
        valid = valid[valid > 50]
        valid = list(valid.index)

        df_train[col] = np.where(df_train[col].isin(valid), df_train[col], "others")
        df_test[col] = np.where(df_test[col].isin(valid), df_test[col], "others")

    for f in cat_feats:
        # print(f)
        df_train[f] = df_train[f].fillna("nan").astype(str)
        df_test[f] = df_test[f].fillna("nan").astype(str)
        le = LabelEncoder().fit(df_train[f].append(df_test[f]))
        df_train[f] = le.transform(df_train[f])
        df_test[f] = le.transform(df_test[f])

    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    print(df_test)
    df_pred_test[id_col] = df_test[id_col]
    df_result = pd.DataFrame()
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train)):
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")

        X_train = df_train.drop([id_col, target_col], axis=1).iloc[train_idx].replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
        y_train = df_train[target_col].iloc[train_idx].values
        X_val = df_train.drop([id_col, target_col], axis=1).iloc[val_idx].replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
        y_val = df_train[target_col].iloc[val_idx].values
        X_test = df_test.drop([id_col], axis=1).replace(np.inf, 0).replace(-np.inf, 0).fillna(0)

        model = get_model(model_name=model_name, df=X_train, cat_feats=cat_feats)

        X_train = get_keras_data(X_train, cat_feats)
        X_val = get_keras_data(X_val, cat_feats)
        X_test = get_keras_data(X_test, cat_feats)

        model.fit(X_train, y_train,
                  epochs=40, batch_size=2048,
                  validation_data=(X_val, y_val),
                  verbose=True,
                  callbacks=[roc_callback(training_data=(X_train, y_train), validation_data=(X_val, y_val)),
                             ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
                             EarlyStopping(monitor="val_loss", patience=6, verbose=1),
                             # CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=10,
                             #          mode="triangular2"),
                             ModelCheckpoint("{}/best.hdf5".format(output_dir), monitor="val_loss", verbose=1,
                                             save_best_only=True)
                             ]
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
    main("basic_wodori")