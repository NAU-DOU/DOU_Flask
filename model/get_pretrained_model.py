import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import sentencepiece as spm

from . import tokenization_kobert
from transformers import BertTokenizer, BertForSequenceClassification, PreTrainedTokenizer, TFBertModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEQ_LEN = 64
BATCH_SIZE = 32

DATA_COLUMN = "발화문"
LABEL_COLUMN = "상황"

tokenizer = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')  # pretrained 모델을 불러오기 (kobert의 monologg)

def get_csv_data():
    csv_data = pd.read_csv('./instance/감정분류데이터셋.csv', encoding='cp949')

    csv_data.loc[(csv_data['상황']=="happiness"), '상황'] = 0
    csv_data.loc[(csv_data['상황']=="surprise"), '상황'] = 1
    csv_data.loc[(csv_data['상황']=="neutral"), '상황'] = 2
    csv_data.loc[(csv_data['상황']=="sadness"), '상황'] = 3
    csv_data.loc[(csv_data['상황']=="disgust"), '상황'] = 4
    csv_data.loc[(csv_data['상황']=="angry"), '상황'] = 5
    csv_data.loc[(csv_data['상황']=="fear"), '상황'] = 6

    return csv_data

def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []

    for i in tqdm(range(len(data_df))):

        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x

def sentence_convert_data(data):
    global tokenizer
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, truncation=True, padding='max_length')

    num_zeros = token.count(0)
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]