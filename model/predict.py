import tensorflow as tf
import numpy as np
import os

from model.get_pretrained_model import sentence_convert_data
from transformers import TFBertModel

loaded_model = tf.keras.models.load_model(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + '/instance/st_model_240420.h5',
    compile=False, custom_objects={'TFBertModel': TFBertModel})

def evaluation_predict(sentence):
    data_x = sentence_convert_data(sentence)
    predict = loaded_model.predict(data_x)
    predict_value = np.ravel(predict)
    predict_answer = np.argmax(predict_value).item()

    result = {
        'sentence': sentence,
        'sentiment': predict_answer,
        'classes': predict_value.tolist()
    }

    return result