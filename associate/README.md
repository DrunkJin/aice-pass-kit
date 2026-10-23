# AICE Associate 합격 패키지

> ⚠️ KT 공식 자료가 아니에요. AICE를 준비하며 만든 개인 학습 자료이고, 실습·모의고사는 실제 시험 문제가 아니에요.

KT AICE **Associate** 시험(Python 실기 14문항 · 90분 · 80점 이상 합격)을 준비하는 자료 모음이에요.
개념 슬라이드로 이해하고, 빈칸 실습 노트북으로 손에 익히고, 자동 채점되는 모의고사 3회로 실전 감각을 만들어요.

> **시작은 `index.html`** 을 브라우저로 여세요. 모든 자료가 학습 순서대로 연결돼 있고, 진도 체크도 할 수 있어요.

## 폴더 구성

```
associate/
├─ index.html              학습 홈 (여기서 시작)
├─ 치트시트.html             A4 가로 4장 요약 (인쇄 / PDF 저장 버튼)
├─ slides/                 개념 슬라이드 (브라우저로 열기, ←/→ 로 넘기기)
│   ├─ 00_시험안내_로드맵.html
│   ├─ 01_파이썬_판다스_기초.html
│   ├─ 02_EDA_시각화.html
│   ├─ 03_데이터_전처리.html
│   ├─ 04_머신러닝.html
│   ├─ 05_딥러닝.html            (PyTorch)
│   └─ 06_실전전략_체크리스트.html
├─ notebooks/
│   ├─ 00_환경확인.ipynb ~ 05_딥러닝_실습.ipynb    빈칸(___) 채우기 실습 + 자동 채점
│   ├─ 모의고사1_배달시간예측_회귀.ipynb             14문항 · 100점 만점 자동 채점
│   ├─ 모의고사2_고객이탈예측_이진분류.ipynb
│   ├─ 모의고사3_와인등급예측_다중분류.ipynb
│   ├─ aice_check.py, _exam_grader.py             채점 헬퍼 (수정하지 마세요)
│   └─ 정답/                                       정답 + 해설 노트북
└─ data/                   실습·모의고사 데이터 (연습용 합성 데이터)
    ├─ _generate_practice_data.py               실습 데이터 재생성 스크립트
    └─ _generate_datasets.py                    모의고사 데이터 재생성 스크립트
```

## 실행 방법

```bash
git clone https://github.com/DrunkJin/aice-pass-kit.git
cd aice-pass-kit/associate
pip install -r requirements.txt      # conda/venv 같은 가상환경 사용 권장
jupyter lab
```

- 처음에는 `notebooks/00_환경확인.ipynb`를 열어 모든 셀을 실행하세요. 라이브러리 버전이 출력되면 준비가 끝난 거예요.
- VS Code에서는 `.ipynb`를 열고 오른쪽 위 **커널 선택**에서 라이브러리를 설치한 Python을 고르세요.
- 슬라이드와 학습 홈은 설치 없이 웹에서도 볼 수 있어요: https://drunkjin.github.io/aice-pass-kit/associate/

검증한 환경: Python 3.10, numpy 1.26.4, pandas 2.2.0, scikit-learn 1.7.2, seaborn 0.13.2, torch 2.10, imbalanced-learn 0.14, xgboost 3.1, lightgbm 4.6

## 공부하는 방법

1. **슬라이드** (15분): ←/→ 키로 넘기고, Q 카드는 정답을 보기 전에 먼저 답해 보세요.
2. **실습 노트북** (40분): `___` 빈칸을 채우고 바로 아래 채점 셀을 실행하면 ✅ 또는 ❌와 힌트가 나와요.
3. **정답 비교** (10분): `notebooks/정답/`에서 해설과 "다른 정답"을 확인해요.
4. **다음 날 복습**: 빈칸 없이 처음부터 다시 작성해 보세요.
5. **모의고사**: 90분 타이머를 켜고 풀어요. 마지막 셀을 실행하면 문항별 채점 결과와 총점이 나와요. 목표는 **3회 모두 90점 이상**이에요.

데이터가 꼬이면 **Kernel → Restart Kernel and Run All Cells**로 처음부터 다시 실행하세요.

## 시험 핵심 정보 (2025년 공개 안내 기준)

| 항목 | 내용 |
|---|---|
| 문항 | 14문항, 모두 Python 코드 작성 실기 |
| 시간 / 합격 | 90분 / 80점 이상 |
| 환경 | KT 클라우드 Jupyter. 딥러닝은 이 패키지에서 PyTorch로 연습해요 |
| 오픈북 | 시험지에 안내된 **공식 문서 사이트 7곳만** 허용. 생성형 AI와 일반 검색은 금지(2025년 2회부터) |
| 응시료 / 유효기간 | 80,000원 / 3년 (2024.12 국가공인 민간자격) |

응시 전에 [aice.study](https://aice.study)에서 최신 규정을 꼭 확인하세요.

## 참고

- 학습 흐름과 예제 구성은 [venture21/AICE-associate](https://github.com/venture21/AICE-associate)의 전처리·머신러닝·딥러닝 예제와 요약 노트를 참고했어요.
- 실습과 모의고사 데이터는 연습용으로 만든 합성 데이터예요. 실제 시험 데이터가 아니에요.
- 슬라이드에서 `E` 키를 누르면 텍스트를 직접 고칠 수 있는 편집 모드가 켜져요. 고친 내용은 브라우저에 저장되고, `Ctrl+S`(⌘+S)를 누르면 HTML 파일로 내려받아요.
