import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

'''
아래 코드는 f1-score들을 정의한 함수.
tensorflow2.0 버전 이후로 f1-score 지원이 metrics애서 사라졌다.
'''

# input값은 one-hot encoding된 것으로 받는다.
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn,axis=0) # element-wise 곱
    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn,axis=0)

    numerator =  tf.dtypes.cast(count_true_positive,tf.float32)
    denominator = tf.dtypes.cast(count_true_positive_false_negative,tf.float32) + K.epsilon()
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    # Recall =  (True Positive) / (True Positive + False Negative)
    recall = numerator / denominator

    return recall

def precision(y_target, y_pred):

    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn,axis=0)
    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn,axis=0)
    # Precision = (True Positive) / (True Positive + False Positive)

    numerator =  tf.dtypes.cast(count_true_positive,tf.float32)
    denominator = tf.dtypes.cast(count_true_positive_false_positive,tf.float32) + K.epsilon()

    precision = numerator / denominator

    return precision

def macro_f1score(y_target, y_pred):
    temp_recall = recall(y_target, y_pred)
    temp_precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    temp_f1score = (2 * temp_recall * temp_precision) / (temp_recall + temp_precision + K.epsilon())

    macro = K.mean(temp_f1score)
    return macro


def weighted_f1score(y_target, y_pred):
    temp_recall = recall(y_target, y_pred)
    temp_precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    temp_f1score = (2 * temp_recall * temp_precision) / (temp_recall + temp_precision + K.epsilon())
    temp_weighted = K.mean(y_target,axis=0)

    weighted = K.mean(temp_f1score * temp_weighted)
    return weighted

