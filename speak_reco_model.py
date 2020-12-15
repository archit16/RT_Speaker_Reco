import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import shutil
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix
import os


def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 16,
    num_epochs = num_epochs,
    shuffle = True,
    num_threads = 1,
    queue_capacity = 1000
  )


def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 4,
    num_epochs = num_epochs,
    shuffle = False,    
    queue_capacity = 1000,
    num_threads = 1
  )

DF_COLUMNS = ['Speaker', 'feat_1', 'feat_2', 'feat_3', 'feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13']
FEATURES = DF_COLUMNS[1:len(DF_COLUMNS)]
LABEL = DF_COLUMNS[0]
df_train = pd.read_csv('speak_reco_multireg.csv', names=DF_COLUMNS, header=None, skiprows=1)
df_valid = pd.read_csv('speak_reco_multireg_valid.csv', names=DF_COLUMNS, header=None, skiprows=1)

def make_feature_cols():
    input_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    return input_cols

  


def print_rmse (model, name, df):
  metrics = model.evaluate(input_fn = make_input_fn(df, 1))
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))

  


OUTDIR = 'speak_reco_multireg_relu_dropout_sparse'



tf.logging.set_verbosity(tf.logging.INFO)
#shutil.rmtree(OUTDIR, ignore_errors = True)



model = tf.estimator.DNNClassifier(
  hidden_units = [1000, 700, 400,100],
  feature_columns = make_feature_cols(),
  model_dir = OUTDIR,
  n_classes = 15,
  activation_fn=tf.nn.relu,
  dropout=0.5
  )


model.train(input_fn = make_input_fn(df_train, num_epochs = 1000))



print_rmse(model, 'validation', df_valid)

predictions = list(model.predict(input_fn = make_prediction_input_fn(df_valid, 1)))
predicted_classes = [p["classes"] for p in predictions]
#print int(predicted_classes[0])

pred=np.empty((0, 49))
for i in predicted_classes:
  pred=np.append(pred, float(i))
print pred

#print df_valid[LABEL]
con_mat= tf.confusion_matrix(df_valid[LABEL], pred, 15)

with tf.Session():
  cm=tf.Tensor.eval(con_mat,feed_dict=None, session=None)
print cm

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False)
plt.show()
  #print('ROC AUC: ', tf.Tensor.eval(fpr, feed_dict=None, session=None))
