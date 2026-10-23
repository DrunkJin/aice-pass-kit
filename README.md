# AICE Pass Kit

KT **AICE** 자격시험을 준비하는 무료 학습 자료예요.
개념 슬라이드로 이해하고, 빈칸 실습 노트북으로 손에 익히고, 자동 채점되는 모의고사로 실전 감각을 만들어요.

> ⚠️ KT 공식 자료가 아니에요. AICE를 준비하며 만든 개인 학습 자료이고, 실습·모의고사 데이터는 연습용 합성 데이터예요.
> 응시 전 [aice.study](https://aice.study)에서 최신 규정을 꼭 확인하세요.

**웹에서 바로 보기:** https://drunkjin.github.io/aice-pass-kit/

## 레벨

| 레벨 | 상태 | 내용 |
|---|---|---|
| [Associate](associate/) | ✅ 공개 | Python 실기 14문항 · 90분 · 80점 합격. 슬라이드 7개(163장), 빈칸 실습 노트북 6개, 자동 채점 모의고사 3회, 치트시트 |
| Professional | 🛠 준비 중 | 비정형 데이터(이미지·텍스트), 성능 최적화 |

## 빠른 시작 (Associate)

```bash
git clone https://github.com/DrunkJin/aice-pass-kit.git
cd aice-pass-kit/associate
pip install -r requirements.txt      # conda/venv 같은 가상환경 사용 권장
jupyter lab
```

슬라이드와 학습 홈은 설치 없이 [웹](https://drunkjin.github.io/aice-pass-kit/associate/)에서 볼 수 있어요.
자세한 사용법은 [associate/README.md](associate/README.md)를 보세요.

## Kaggle에서 바로 실행하기

설치 없이 브라우저에서 풀어 볼 수 있어요. 문제판은 **Copy & Edit**로 복사한 뒤 답안을 쓰고, 맨 아래 채점 셀을 실행하세요.

| 노트북 | 문제 | 정답·해설 |
|---|---|---|
| 모의고사 1 · 배달시간 예측 (회귀) | [풀기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-1-regression) | [보기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-1-regression-answer) |
| 모의고사 2 · 고객 이탈 예측 (이진분류) | [풀기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-2-binary-classification) | [보기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-2-binary-classification-answer) |
| 모의고사 3 · 와인 등급 예측 (다중분류) | [풀기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-3-multiclass) | [보기](https://www.kaggle.com/code/cocoyachi/aice-mock-exam-3-multiclass-answer) |
| 실습 01 · 판다스 기초 | [빈칸판 (GitHub)](https://github.com/DrunkJin/aice-pass-kit/blob/main/associate/notebooks/01_%ED%8C%90%EB%8B%A4%EC%8A%A4_%EA%B8%B0%EC%B4%88_%EC%8B%A4%EC%8A%B5.ipynb) | [보기](https://www.kaggle.com/code/cocoyachi/aice-practice-01-pandas-basics-answer) |
| 실습 02 · EDA·시각화 | [빈칸판 (GitHub)](https://github.com/DrunkJin/aice-pass-kit/blob/main/associate/notebooks/02_EDA_%EC%8B%9C%EA%B0%81%ED%99%94_%EC%8B%A4%EC%8A%B5.ipynb) | [보기](https://www.kaggle.com/code/cocoyachi/aice-practice-02-eda-visualization-answer) |
| 실습 03 · 전처리 | [빈칸판 (GitHub)](https://github.com/DrunkJin/aice-pass-kit/blob/main/associate/notebooks/03_%EC%A0%84%EC%B2%98%EB%A6%AC_%EC%8B%A4%EC%8A%B5.ipynb) | [보기](https://www.kaggle.com/code/cocoyachi/aice-practice-03-preprocessing-answer) |
| 실습 04 · 머신러닝 | [빈칸판 (GitHub)](https://github.com/DrunkJin/aice-pass-kit/blob/main/associate/notebooks/04_%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EC%8B%A4%EC%8A%B5.ipynb) | [보기](https://www.kaggle.com/code/cocoyachi/aice-practice-04-machine-learning-answer) |
| 실습 05 · 딥러닝 (PyTorch) | [빈칸판 (GitHub)](https://github.com/DrunkJin/aice-pass-kit/blob/main/associate/notebooks/05_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EC%8B%A4%EC%8A%B5.ipynb) | [보기](https://www.kaggle.com/code/cocoyachi/aice-practice-05-deep-learning-pytorch-answer) |

데이터와 채점기: [Kaggle Dataset · aice-pass-kit-data](https://www.kaggle.com/datasets/cocoyachi/aice-pass-kit-data)

## 구성

```
aice-pass-kit/
├─ index.html          레벨 선택 페이지
├─ associate/          Associate 패키지
│   ├─ index.html      학습 홈 (진도 체크)
│   ├─ slides/         개념 슬라이드 00~06
│   ├─ notebooks/      실습 00~05 · 모의고사 1~3 · 정답/
│   ├─ data/           연습용 합성 데이터 + 생성 스크립트
│   └─ 치트시트.html     A4 가로 4장 요약
└─ professional/       (준비 중)
```

## 참고

- 학습 흐름은 [venture21/AICE-associate](https://github.com/venture21/AICE-associate)의 예제 구성을 참고했어요. 설명, 코드, 데이터는 이 저장소에서 새로 작성했어요.
- 오타나 틀린 내용을 발견하면 Issue로 알려 주세요.

## 라이선스

- 슬라이드, 노트북 설명, 문서: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko). 출처를 표시하면 자유롭게 쓰고 고칠 수 있어요.
- 코드(`.py`, 노트북의 코드 셀): MIT

자세한 내용은 [LICENSE](LICENSE)를 보세요.
