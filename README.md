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
