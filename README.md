# DOU Flask ML Serving API Server
AI 감정 분석 도우미 서비스, DOU의 감정 분석 ML 서버 레포지토리

PyTorch로 작성된 모델이며, 사용하기에 Python 기반 프레임워크를 사용하는 것이 편리할거라 생각하여, (GCP 프리 티어 사용을 통한 금액 절감도 이유에 존재)<br>
이와 같이 Flask로 ML Serving 서버를 따로 만들게 되었다.

[📽️ DOU 시연 영상](https://drive.google.com/file/d/1hSAbpaFYzKeULfNx4ukcbTy0-Qf63d1q/view)

[💻 DOU 발표 자료](https://drive.google.com/file/d/1hfUSxAX4w-U8A6k6vghWjUectPvPycOS/view)

## 📋 프로젝트 개요
한국어 음성 기반 감정 분석을 위한 KoBERT 기반 감정 분류 모델을 서빙하는 Flask API 서버입니다.

7가지 감정(기쁨, 슬픔, 분노, 불안, 놀람, 꺼림, 중립)을 분류하고, DOU Backend 서버에 REST API를 통해 분석 결과를 제공합니다.

## 🙋 팀 구성
| 이름 | 깃허브 | 역할 |
| --- | --- | --- |
| 한상은 | [@silvarge](https://github.com/silvarge) | 팀장 / 기획, 백엔드 및 ML 전체 담당 |
| 김가현 | [@gahyunkim](https://github.com/gahyunkim) | 기획, 프론트엔드(안드로이드) 및 디자인 담당 |

## 🛠️ 기술 스택
| 분류 | 내용 |
| --- | --- |
| Framework | Flask (Python) |
| ML Framework | PyTorch |
| Pre-trained Model | KoBERT |
| Deploy & Infra | GitHub Actions, Docker, Google Cloud Platform (GCP VM) |
| Others | Transformers, NumPy, Pandas, scikit-learn | 

## 🚀 주요 기능
### 감정 분석 모델
- KoBERT 기반 한국어 감정 분류: 7가지 감정 카테고리 분류 (기쁨, 슬픔, 분노, 불안, 놀람, 꺼림, 중립)
- 커스텀 학습: 한국어 감정 데이터셋으로 파인튜닝
- 고성능 추론: 평균 F1-score 92% 달성
- 실시간 분석: REST API를 통한 실시간 감정 분석

### API 서비스
- 감정 분석 API 제공: 텍스트 입력 시 감정 분류 및 신뢰도 반환
- 모델 상태 확인: 모델 로딩 상태 및 서버 헬스체크
- 배치 처리: 다중 텍스트 일괄 분석 지원

## 🐛 문제 해결 및 개선사항
### 해결된 주요 이슈
- Keras 기반 모델 개선 작업 중 Keras 버전 업데이트 및 monologg/kobert 업데이트로 인한 호환성 오류 발생
  - KerasTensor → Tensor 변환 코드를 작성해도 지속적으로 오류가 발생하여, 제한된 시간 내 안정적 개선을 위해 PyTorch 기반 KoBERT로 모델 재설계
  - 재설계 후, 기존에 정확도가 낮았던 특정 감정(중립, 꺼림, 분노) 분류 성능이 크게 개선됨 (f1-score 기준 60% 대 -> 80% 대)
- 하이퍼파라미터 최적화 부족
  - 기존에는 튜닝이 미흡하여 성능 편차 발생
  - Grid Search를 활용해 Learning Rate, Batch Size, Dropout 비율을 체계적으로 탐색
  - 모든 감정 카테고리에서 f1-score 90% 이상 달성

## 🧠 모델 상세
### KoBERT 기반 감정 분류 모델
<img width="1553" height="199" alt="image" src="https://github.com/user-attachments/assets/aadf5044-154c-4922-b15a-b12d575f01ba" />

### 하이퍼 파라미터
- Learning Rate: 5e-5
- Batch Size: 32
- Max Length: 128
- Dropout Rate: 0.55
- Epochs: 4
- Optimizer: AdamW

## 인프라 구조
<img width="374" height="275" alt="image" src="https://github.com/user-attachments/assets/41c1cae5-7a5d-4412-ab71-95220cd41c88" />

## 📂 프로젝트 구조
``` bash
DOU_Backend/
├── .github/workflows/     # GitHub Actions CI/CD
├── .platform/nginx/       # Elastic Beanstalk 설정
├── src/
│   ├── apis/
│   │   ├── auths/         # 인증/인가 (카카오 OAuth, JWT)
│   │   ├── sentiments/    # 감정 분석 API 연동
│   │   ├── records/       # 대화 기록 및 감정 히스토리
│   │   ├── gpt/          # GPT 응답 처리
│   │   └── users/        # 사용자 관리
│   ├── commons/          # 공통 모듈 (예외처리, 로거, Swagger)
│   └── main.ts
├── test/                 # E2E 테스트
└── package.json
```
