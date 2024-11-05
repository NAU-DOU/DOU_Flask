from flask import request, jsonify
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

from model.storage import download_file

# BERTClassifier 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.55):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = outputs.pooler_output  # BERT의 pooler_output을 사용
        out = self.dropout(pooler_output)
        return self.classifier(out)

# 모델 및 토크나이저 로드
model_path = download_file('dou-flask', 'sentiment_model.pth')
device = torch.device("cpu")  # GCP에서 CPU를 사용할 경우
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
bert_model = AutoModel.from_pretrained('monologg/kobert', trust_remote_code=True)
model = BERTClassifier(bert_model)  # BERTClassifier 인스턴스 생성
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 감정 레이블 정의
emotion_labels = ["행복", "놀람", "중립", "슬픔", "꺼림", "분노", "두려움"]

def preprocess_text(text):
    """텍스트를 토큰화하여 모델 입력 형식으로 변환"""
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    return input_ids, attention_mask, token_type_ids

def predict(sentence):
    """텍스트 감정 예측 엔드포인트"""
    
    # 입력 데이터 전처리
    input_ids, attention_mask, token_type_ids = preprocess_text(sentence)
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        logits = outputs[0].cpu().numpy()
        pred_label_idx = np.argmax(logits)
        pred_emotion = emotion_labels[pred_label_idx]
    
    # 결과 반환
    return {
        'sentence': sentence,
        'sentiment': pred_emotion,
        'sentiment_idx': int(pred_label_idx)
      }