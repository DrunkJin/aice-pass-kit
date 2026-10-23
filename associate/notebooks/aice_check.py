"""AICE Associate 실습 자동 체크 헬퍼.

노트북에서 이렇게 사용합니다:
    import aice_check as check
    ...
    check.p01_q3()      # 01 노트북 3번 문제 확인 → ✅ 또는 ❌ + 힌트
    check.progress()    # 지금까지 푼 문제 현황

- 인자를 넘길 필요가 없습니다. 노트북에 만들어진 변수(df, X_train ...)를 이름으로 찾아서 확인합니다.
  → 그래서 문제에서 정한 "변수명"을 정확히 지켜야 합니다. (실제 시험도 똑같습니다!)
- 어떤 경우에도 에러로 멈추지 않고 ❌ 메시지와 힌트를 보여줍니다.
"""
import functools
import inspect
import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, '..', 'data')
if not os.path.isdir(_DATA):          # Kaggle 등: 데이터와 채점기가 같은 폴더에 있을 때
    _DATA = _HERE
_RESULTS = {}          # {'p01': {3: True, ...}}


# ---------------------------------------------------------------------------
# 내부 도구
# ---------------------------------------------------------------------------
class _Fail(Exception):
    pass


class _Missing(Exception):
    def __init__(self, name):
        super().__init__(name)
        self.name = name


class _NS:
    """노트북 변수에 v.df 처럼 접근. 없으면 _Missing 발생."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._d:
            return self._d[name]
        raise _Missing(name)

    def has(self, name):
        return name in self._d


def need(cond, msg):
    if not cond:
        raise _Fail(msg)


def _close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except Exception:
        return False


def _q(nb, no, title):
    """체크 함수 데코레이터: 호출한 셀의 전역 변수를 찾아 검사하고 결과를 출력."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ns=None):
            if ns is None:
                ns = inspect.currentframe().f_back.f_globals
            try:
                msg = fn(_NS(ns))
                ok = True
            except _Missing as e:
                ok, msg = False, (f"변수 `{e.name}` 이(가) 아직 없어요. 답안 셀을 실행했는지, "
                                  f"변수명 철자(대소문자 포함)가 문제와 같은지 확인하세요.")
            except _Fail as e:
                ok, msg = False, str(e)
            except Exception as e:  # 어떤 경우에도 노트북을 멈추지 않음
                ok, msg = False, f"확인 중 문제가 생겼어요 → {type(e).__name__}: {e}"
            _RESULTS.setdefault(nb, {})[no] = ok
            head = f"✅ [{nb.upper()} 문제 {no}] {title} — 정답!" if ok else f"❌ [{nb.upper()} 문제 {no}] {title}"
            print(head)
            if msg:
                print('   ' + str(msg).replace('\n', '\n   '))
        return wrapper
    return deco


def progress():
    """지금까지 확인한 문제 현황 출력."""
    if not _RESULTS:
        print('아직 확인한 문제가 없어요.')
        return
    for nb, res in sorted(_RESULTS.items()):
        done = sum(res.values())
        bad = [k for k, v in sorted(res.items()) if not v]
        bar = '■' * done + '□' * (len(res) - done)
        print(f"{nb.upper()}: {done}/{len(res)} 정답 {bar}" + (f"  (다시 볼 문제: {bad})" if bad else ''))


@functools.lru_cache(maxsize=None)
def _raw(name):
    path = os.path.join(_DATA, name)
    if name.endswith('.json'):
        return pd.read_json(path)
    return pd.read_csv(path)


def telco():
    return _raw('telco_churn.csv').copy()


def house():
    return _raw('house_price.csv').copy()


def _is_module(obj, modname):
    return inspect.ismodule(obj) and obj.__name__.split('.')[0] == modname


def _axes(obj, what='ax'):
    import matplotlib.axes
    need(isinstance(obj, matplotlib.axes.Axes),
         f"`{what}` 에는 그래프 함수가 돌려주는 Axes 가 들어 있어야 해요. 예) {what} = sns.countplot(...)")
    return obj


# ===========================================================================
# 00 환경 확인
# ===========================================================================
@_q('p00', 1, 'pandas / numpy import')
def p00_q1(v):
    need(_is_module(v.pd, 'pandas'), '`pd` 가 pandas 가 아니에요 → import pandas as pd')
    need(_is_module(v.np, 'numpy'), '`np` 가 numpy 가 아니에요 → import numpy as np')
    return f'pandas {v.pd.__version__}, numpy {v.np.__version__} 준비 완료.'


@_q('p00', 2, '버전 문자열 저장')
def p00_q2(v):
    need(v.pd_version == pd.__version__, '`pd_version` 에 pd.__version__ 값을 저장하세요.')
    return ''


@_q('p00', 3, '데이터 폴더 확인')
def p00_q3(v):
    files = v.data_files
    need(isinstance(files, (list, tuple)), '`data_files` 는 os.listdir(DATA_DIR) 의 결과(리스트)여야 해요.')
    need('telco_churn.csv' in files, 'telco_churn.csv 가 목록에 없어요. DATA_DIR 경로를 확인하세요.')
    return f'데이터 파일 {len([f for f in files if not f.startswith("_")])}개 확인.'


# ===========================================================================
# 01 판다스 기초 (telco_churn.csv, customer_info.json, customer_usage.csv)
# ===========================================================================
@_q('p01', 1, 'import 별칭')
def p01_q1(v):
    need(_is_module(v.pd, 'pandas'), 'pandas 는 `pd` 라는 별칭으로 import 하세요.')
    need(_is_module(v.np, 'numpy'), 'numpy 는 `np` 라는 별칭으로 import 하세요.')
    return ''


@_q('p01', 2, 'CSV 읽기')
def p01_q2(v):
    df = v.df
    need(isinstance(df, pd.DataFrame), '`df` 가 DataFrame 이 아니에요. pd.read_csv(...) 결과를 저장하세요.')
    need(len(df) == 3000 and 'customerID' in df.columns, 'telco_churn.csv 를 읽었나요? (3000행이어야 해요)')
    return f'{df.shape[0]}행 × {df.shape[1]}열'


@_q('p01', 3, 'shape')
def p01_q3(v):
    need((v.n_rows, v.n_cols) == (3000, 18), 'df.shape 는 (행, 열) 튜플이에요 → n_rows, n_cols = df.shape')
    return ''


@_q('p01', 4, '결측치 개수')
def p01_q4(v):
    s = v.null_counts
    need(isinstance(s, pd.Series), '`null_counts` 는 Series 여야 해요 → df.isnull().sum()')
    need(s.equals(telco().isnull().sum()), '컬럼별 결측치 개수와 달라요. df.isnull().sum() (또는 isna) 를 쓰세요.')
    return f"결측이 가장 많은 컬럼: {s.idxmax()} ({s.max()}개) → 전처리에서 삭제 후보!"


@_q('p01', 5, '값 분포 value_counts')
def p01_q5(v):
    need(v.churn_counts.equals(telco()['Churn'].value_counts()), "df['Churn'].value_counts() 를 저장하세요.")
    vc = v.churn_counts
    return f"Yes 비율 {vc['Yes'] / vc.sum():.1%} → 불균형 데이터예요."


@_q('p01', 6, '고유값 개수 nunique')
def p01_q6(v):
    need(v.n_payment == 4, "df['PaymentMethod'].nunique() 는 4 예요.")
    return ''


@_q('p01', 7, '여러 컬럼 선택')
def p01_q7(v):
    need(isinstance(v.sub, pd.DataFrame), '대괄호 두 겹 [[ ]] 을 써야 DataFrame 이 돼요.')
    need(list(v.sub.columns) == ['customerID', 'tenure', 'Churn'], "컬럼 순서/이름: ['customerID', 'tenure', 'Churn']")
    return ''


@_q('p01', 8, 'loc (라벨 기준, 끝 포함)')
def p01_q8(v):
    s = v.first3_tenure
    need(len(s) == 3 and list(s.index) == [0, 1, 2], "df.loc[0:2, 'tenure'] → loc 는 끝(2)을 포함해서 3개!")
    return ''


@_q('p01', 9, 'iloc (위치 기준, 끝 미포함)')
def p01_q9(v):
    need(v.block.shape == (5, 3), 'df.iloc[0:5, 0:3] → iloc 는 끝을 포함하지 않아서 5행 × 3열.')
    need(list(v.block.columns) == list(telco().columns[:3]), '앞에서부터 3개 컬럼이어야 해요.')
    return ''


@_q('p01', 10, '조건 필터링')
def p01_q10(v):
    t = telco()
    need(len(v.long_customers) == (t['tenure'] >= 60).sum() and (v.long_customers['tenure'] >= 60).all(),
         "df[df['tenure'] >= 60]")
    return f'{len(v.long_customers)}명'


@_q('p01', 11, '여러 조건 & ')
def p01_q11(v):
    t = telco()
    exp = ((t['Contract'] == 'Month-to-month') & (t['tenure'] < 12)).sum()
    need(len(v.risky) == exp, "(조건1) & (조건2) — 조건마다 괄호 필수! and 가 아니라 & 를 써요.")
    return f'{exp}명'


@_q('p01', 12, 'groupby 평균')
def p01_q12(v):
    exp = telco().groupby('Contract')['MonthlyCharges'].mean()
    need(isinstance(v.mean_charge, pd.Series), "df.groupby('Contract')['MonthlyCharges'].mean()")
    need(np.allclose(v.mean_charge.sort_index().values, exp.sort_index().values), '계약 형태별 MonthlyCharges 평균과 달라요.')
    return ''


@_q('p01', 13, '정렬 sort_values')
def p01_q13(v):
    exp = telco().sort_values('tenure', ascending=False)['tenure'].head(10).tolist()
    need(len(v.top_tenure) == 10, 'head(10) 으로 10개만!')
    need(v.top_tenure['tenure'].tolist() == exp, 'ascending=False 로 내림차순 정렬하세요.')
    return ''


@_q('p01', 14, '컬럼 삭제 drop')
def p01_q14(v):
    need('customerID' not in v.df2.columns, "df.drop('customerID', axis=1) — axis=1 은 '컬럼' 방향")
    need('customerID' in v.df.columns, '원본 df 는 그대로 둬야 해요 (결과는 df2 에만).')
    return ''


@_q('p01', 15, '문자열 → 숫자 변환')
def p01_q15(v):
    col = v.df['TotalCharges']
    need(col.dtype.kind == 'f', "TotalCharges 가 아직 숫자가 아니에요. replace(' ', '0') 후 astype(float)")
    need(col.isna().sum() == 0, '결측(NaN)이 남아 있어요.')
    return f"합계 {col.sum():,.0f}"


@_q('p01', 16, 'apply + lambda')
def p01_q16(v):
    need('tenure_year' in v.df.columns, "df['tenure_year'] 컬럼을 만드세요.")
    need((v.df['tenure_year'] == v.df['tenure'] // 12).all(), 'lambda x: x // 12  (몫 연산자 //)')
    return ''


@_q('p01', 17, 'map 으로 값 바꾸기')
def p01_q17(v):
    need('Churn_num' in v.df.columns, "df['Churn_num'] 컬럼을 만드세요.")
    need(set(v.df['Churn_num'].unique()) == {0, 1}, "map({'Yes': 1, 'No': 0})")
    need(v.df['Churn_num'].sum() == (telco()['Churn'] == 'Yes').sum(), "Yes → 1, No → 0 이어야 해요.")
    return ''


@_q('p01', 18, 'JSON / CSV 읽기')
def p01_q18(v):
    need(isinstance(v.info, pd.DataFrame) and v.info.shape == (500, 4), "info = pd.read_json(DATA_DIR + 'customer_info.json')")
    need(isinstance(v.usage, pd.DataFrame) and v.usage.shape == (500, 4), "usage = pd.read_csv(DATA_DIR + 'customer_usage.csv')")
    return ''


@_q('p01', 19, 'merge (inner)')
def p01_q19(v):
    need(v.merged.shape == (480, 7), "pd.merge(info, usage, on='customerID', how='inner') → 양쪽에 다 있는 480명")
    return ''


@_q('p01', 20, 'merge (left)')
def p01_q20(v):
    need(v.merged_left.shape == (500, 7), "how='left' → 왼쪽(info) 500명 전부 유지")
    need(v.merged_left['data_gb'].isna().sum() == 20, '오른쪽에 없는 20명은 NaN 이 돼요.')
    return '왼쪽에만 있는 20명의 사용량 컬럼은 NaN!'


@_q('p01', 21, 'concat')
def p01_q21(v):
    need(len(v.stacked) == 200, 'head(100) 과 tail(100) 을 위아래로 붙이면 200행')
    need(list(v.stacked.index) == list(range(200)), 'ignore_index=True 로 인덱스를 0~199 로 새로 매기세요.')
    return ''


@_q('p01', 22, 'CSV 저장')
def p01_q22(v):
    need(os.path.exists('merged_result.csv'), "merged.to_csv('merged_result.csv', index=False)")
    back = pd.read_csv('merged_result.csv')
    need('Unnamed: 0' not in back.columns, 'index=False 를 빼먹으면 Unnamed: 0 컬럼이 생겨요.')
    need(len(back) == 480, 'merged (480행) 를 저장하세요.')
    return ''


# ===========================================================================
# 02 EDA / 시각화
# ===========================================================================
@_q('p02', 1, 'matplotlib / seaborn import')
def p02_q1(v):
    need(_is_module(v.plt, 'matplotlib') and v.plt.__name__ == 'matplotlib.pyplot', 'import matplotlib.pyplot as plt')
    need(_is_module(v.sns, 'seaborn'), 'import seaborn as sns')
    return ''


@_q('p02', 2, 'countplot')
def p02_q2(v):
    ax = _axes(v.ax2, 'ax2')
    need(ax.get_xlabel() == 'Churn', "x='Churn' 으로 그려야 해요.")
    return ''


@_q('p02', 3, 'countplot + hue')
def p02_q3(v):
    ax = _axes(v.ax3, 'ax3')
    need(ax.get_xlabel() == 'Contract', "x='Contract'")
    need(ax.get_legend() is not None, "hue='Churn' 을 주면 범례가 생겨요.")
    return ''


@_q('p02', 4, 'histplot')
def p02_q4(v):
    ax = _axes(v.ax4, 'ax4')
    need(ax.get_xlabel() == 'tenure', "x='tenure'")
    need(len(ax.patches) == 30, 'bins=30 으로 막대 30개')
    return ''


@_q('p02', 5, 'kdeplot + hue')
def p02_q5(v):
    ax = _axes(v.ax5, 'ax5')
    need(ax.get_xlabel() == 'tenure', "x='tenure'")
    need(len(ax.lines) + len(ax.collections) >= 2, "hue='Churn' 으로 곡선 2개")
    return ''


@_q('p02', 6, 'boxplot')
def p02_q6(v):
    ax = _axes(v.ax6, 'ax6')
    need(ax.get_ylabel() == 'MonthlyCharges', "y='MonthlyCharges'")
    return ''


@_q('p02', 7, '이상치 개수 세기')
def p02_q7(v):
    need(int(v.n_outliers) == 7, "(df['MonthlyCharges'] > 200).sum() — True 는 1 로 세어져요.")
    return '이 7개는 입력 오류로 보여요 → 전처리에서 제거!'


@_q('p02', 8, 'scatterplot')
def p02_q8(v):
    ax = _axes(v.ax8, 'ax8')
    need(ax.get_xlabel() == 'tenure' and ax.get_ylabel() == 'TotalCharges', "x='tenure', y='TotalCharges'")
    return ''


@_q('p02', 9, 'jointplot')
def p02_q9(v):
    import seaborn as sns
    need(isinstance(v.g9, sns.JointGrid), 'jointplot 은 Axes 가 아니라 JointGrid 를 돌려줘요 → g9 = sns.jointplot(...)')
    need(v.g9.ax_joint.get_xlabel() == 'tenure' and v.g9.ax_joint.get_ylabel() == 'MonthlyCharges',
         "x='tenure', y='MonthlyCharges'")
    return ''


@_q('p02', 10, 'barplot (평균 막대)')
def p02_q10(v):
    ax = _axes(v.ax10, 'ax10')
    need(ax.get_xlabel() == 'Contract' and ax.get_ylabel() == 'Churn_num', "x='Contract', y='Churn_num'")
    return '막대 높이 = 그룹별 평균 = 이탈률'


@_q('p02', 11, '그래프 해석: 이탈률이 가장 높은 계약')
def p02_q11(v):
    t = telco()
    t = t[t['MonthlyCharges'] < 200]
    exp = t.assign(c=(t['Churn'] == 'Yes').astype(int)).groupby('Contract')['c'].mean().idxmax()
    need(v.answer11 == exp, '막대가 가장 높은 계약 형태 이름을 문자열로 그대로 적어보세요.')
    return '월 단위 계약 고객이 가장 많이 떠나요.'


@_q('p02', 12, '상관계수 corr')
def p02_q12(v):
    c = v.corr
    need(isinstance(c, pd.DataFrame) and c.shape == (4, 4), "4개 컬럼만 골라 .corr()")
    need(np.allclose(np.diag(c.values), 1), '대각선은 자기 자신과의 상관계수라 1 이어야 해요.')
    return ''


@_q('p02', 13, 'heatmap (annot)')
def p02_q13(v):
    ax = _axes(v.ax13, 'ax13')
    need(len(ax.texts) >= 16, 'annot=True 로 칸마다 숫자를 표시하세요.')
    return ''


@_q('p02', 14, '그래프 해석: tenure 와 가장 상관 높은 컬럼')
def p02_q14(v):
    exp = v.corr['tenure'].drop('tenure').abs().idxmax()
    need(v.answer14 == exp, "corr['tenure'] 에서 자기 자신(1.0)을 빼고 가장 큰 값의 컬럼명을 적으세요.")
    return f"{exp}: 가입기간이 길수록 누적요금도 커지니 당연하죠."


@_q('p02', 15, 'matplotlib 제목/축 이름')
def p02_q15(v):
    ax = _axes(v.ax15, 'ax15')
    need(ax.get_title() != '', 'plt.title(...) 로 제목을 다세요.')
    need(ax.get_xlabel() != '' and ax.get_ylabel() != '', 'plt.xlabel / plt.ylabel 로 축 이름을 다세요.')
    return ''


@_q('p02', 16, 'subplots 로 나란히 그리기')
def p02_q16(v):
    axes = v.axes
    need(len(axes) == 2, 'plt.subplots(1, 2) → axes 는 2개')
    need(axes[0].get_xlabel() == 'gender' and axes[1].get_xlabel() == 'Partner',
         "ax=axes[0] 에 gender, ax=axes[1] 에 Partner")
    return ''


# ===========================================================================
# 03 전처리
# ===========================================================================
def _p03_after_dropna():
    t = telco().drop('customerID', axis=1)
    t = t[t['MonthlyCharges'] < 200]
    t = t.drop('DeviceProtection', axis=1)
    return t.dropna()


@_q('p03', 1, '불필요한 컬럼 삭제')
def p03_q1(v):
    need('customerID' not in v.df.columns, "df = df.drop('customerID', axis=1)  ← 결과를 다시 df 에 저장!")
    return ''


@_q('p03', 2, '이상치 제거 (조건)')
def p03_q2(v):
    need(v.df['MonthlyCharges'].max() < 200, "df = df[df['MonthlyCharges'] < 200]")
    need(len(v.df) == 2993, '7개 행만 빠져서 2993행이어야 해요.')
    return ''


@_q('p03', 3, '이상치 제거 (IQR)')
def p03_q3(v):
    h = house()
    q1, q3 = h['area_m2'].quantile(0.25), h['area_m2'].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    need(_close(v.q1, q1) and _close(v.q3, q3), "q1 = quantile(0.25), q3 = quantile(0.75)")
    need(_close(v.iqr, iqr), 'iqr = q3 - q1')
    need(_close(v.lower, lo) and _close(v.upper, hi), 'lower = q1 - 1.5*iqr, upper = q3 + 1.5*iqr')
    exp = ((h['area_m2'] >= lo) & (h['area_m2'] <= hi)).sum()
    need(len(v.house_clean) == exp, '(area_m2 >= lower) & (area_m2 <= upper) 인 행만 남기세요.')
    return f'{len(h) - exp}개 행 제거됨'


@_q('p03', 4, '결측치 확인')
def p03_q4(v):
    s = v.null_counts
    need(isinstance(s, pd.Series) and s.get('DeviceProtection', 0) > 1000, 'df.isnull().sum()')
    return 'DeviceProtection 은 절반 가까이 비어 있어요 → 채우기보다 컬럼 삭제!'


@_q('p03', 5, '결측 많은 컬럼 삭제')
def p03_q5(v):
    need('DeviceProtection' not in v.df.columns, "df = df.drop('DeviceProtection', axis=1)")
    return ''


@_q('p03', 6, '결측 행 삭제 dropna')
def p03_q6(v):
    need(v.df.isnull().sum().sum() == 0, 'df = df.dropna()  (axis=0 이 기본값: 행 삭제)')
    need(len(v.df) == len(_p03_after_dropna()), '행 수가 예상과 달라요. 앞 단계를 순서대로 실행했나요?')
    return f'{len(v.df)}행 남음'


@_q('p03', 7, '결측 채우기 (중앙값/평균)')
def p03_q7(v):
    h = v.house
    need(h['built_year'].isna().sum() == 0 and h['station_dist_m'].isna().sum() == 0, 'fillna 후 다시 컬럼에 저장했나요?')
    raw = house()
    filled = h.loc[raw['built_year'].isna(), 'built_year']
    need(np.allclose(filled, raw['built_year'].median()), 'built_year 는 median() 으로 채우세요.')
    filled2 = h.loc[raw['station_dist_m'].isna(), 'station_dist_m']
    need(np.allclose(filled2, raw['station_dist_m'].mean()), 'station_dist_m 는 mean() 으로 채우세요.')
    return ''


@_q('p03', 8, '결측 채우기 (최빈값)')
def p03_q8(v):
    exp = telco()['gender'].mode()[0]
    need(v.gender_mode == exp, "mode() 는 Series 를 돌려줘요 → [0] 으로 첫 값을 꺼내세요.")
    need(v.tmp['gender'].isna().sum() == 0, "tmp['gender'] = tmp['gender'].fillna(gender_mode)")
    return f'최빈값 = {exp}'


@_q('p03', 9, "' ' 공백 → 숫자")
def p03_q9(v):
    col = v.df['TotalCharges']
    need(col.dtype.kind == 'f', "pd.to_numeric(..., errors='coerce') 로 숫자로 바꾸세요.")
    need(col.isna().sum() == 0, '공백이 NaN 으로 바뀌었으니 fillna(0) 까지!')
    return ''


@_q('p03', 10, 'LabelEncoder')
def p03_q10(v):
    need(set(np.unique(v.df['Churn'])) == {0, 1}, "df['Churn'] = le.fit_transform(df['Churn'])")
    need(list(v.le.classes_) == ['No', 'Yes'], 'le.classes_ 는 알파벳 순 → No=0, Yes=1')
    return ''


@_q('p03', 11, 'map 으로 0/1 변환')
def p03_q11(v):
    need(set(v.df['gender'].unique()) == {0, 1}, "map({'Male': 1, 'Female': 0})")
    return ''


@_q('p03', 12, 'astype')
def p03_q12(v):
    need(v.df['SeniorCitizen'].dtype.kind == 'i', "astype(int)")
    return ''


@_q('p03', 13, 'object 컬럼 고르기')
def p03_q13(v):
    exp = list(v.df.select_dtypes('object').columns)
    need(list(v.obj_cols) == exp, "df.select_dtypes('object').columns")
    return f'{len(exp)}개: ' + ', '.join(exp)


@_q('p03', 14, '원-핫 인코딩 get_dummies')
def p03_q14(v):
    d = v.df_preset
    need(isinstance(d, pd.DataFrame), 'pd.get_dummies(...) 결과를 df_preset 에 저장하세요.')
    need(len(d.select_dtypes('object').columns) == 0, '아직 문자열 컬럼이 남아 있어요. columns=obj_cols 를 주세요.')
    need(len(d) == len(v.df), '행 수는 그대로여야 해요.')
    return f'{v.df.shape[1]}개 → {d.shape[1]}개 컬럼'


@_q('p03', 15, 'X / y 나누기')
def p03_q15(v):
    need('Churn' not in v.X.columns, "X = df_preset.drop('Churn', axis=1)")
    need(v.y.name == 'Churn' and len(v.y) == len(v.X), "y = df_preset['Churn']")
    return ''


@_q('p03', 16, 'train_test_split (stratify)')
def p03_q16(v):
    from sklearn.model_selection import train_test_split
    a, b, c, d = train_test_split(v.X, v.y, test_size=0.2, stratify=v.y, random_state=42)
    need(v.X_train.shape == a.shape and v.X_test.shape == b.shape, 'test_size=0.2 → 학습 80% / 검증 20%')
    need(list(v.X_train.index) == list(a.index), 'stratify=y, random_state=42 를 정확히 주세요. 순서: X_train, X_test, y_train, y_test')
    need(list(v.y_test.index) == list(d.index), '반환 순서를 확인하세요: X_train, X_test, y_train, y_test')
    return f'학습 이탈률 {v.y_train.mean():.3f} ≈ 검증 이탈률 {v.y_test.mean():.3f}  ← stratify 효과'


@_q('p03', 17, 'MinMaxScaler')
def p03_q17(v):
    need(np.allclose(v.scaler.data_min_, v.X_train.min().values.astype(float)),
         'scaler 는 X_train 으로만 fit 해야 해요 (fit_transform(X_train)).')
    need(np.isclose(v.X_train_s.min(), 0) and np.isclose(v.X_train_s.max(), 1), '학습 데이터는 0~1 사이가 돼야 해요.')
    need(v.X_test_s.shape == v.X_test.shape, 'X_test_s = scaler.transform(X_test)  ← test 는 transform 만!')
    return ''


@_q('p03', 18, 'StandardScaler')
def p03_q18(v):
    m = np.asarray(v.X_train_z).mean(axis=0)
    need(np.allclose(m, 0, atol=1e-6), 'X_train_z = std_scaler.fit_transform(X_train)')
    need(np.allclose(v.std_scaler.mean_, v.X_train.mean().values.astype(float)), 'X_train 으로만 fit 하세요.')
    need(v.X_test_z.shape == v.X_test.shape, 'X_test_z = std_scaler.transform(X_test)')
    return ''


@_q('p03', 19, '전처리 결과 저장')
def p03_q19(v):
    need(os.path.exists('telco_preprocessed.csv'), "df_preset.to_csv('telco_preprocessed.csv', index=False)")
    back = pd.read_csv('telco_preprocessed.csv')
    need('Unnamed: 0' not in back.columns, 'index=False 를 주세요.')
    return ''


# ===========================================================================
# 04 머신러닝
# ===========================================================================
def _fitted(model):
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def _param(model, name, val, label):
    got = model.get_params().get(name)
    need(got == val, f"{label}: {name}={val!r} 로 만들어야 해요. (지금: {got!r})")


@_q('p04', 1, 'import')
def p04_q1(v):
    from sklearn.linear_model import LogisticRegression
    need(v.LogisticRegression is LogisticRegression, 'from sklearn.linear_model import LogisticRegression')
    return ''


@_q('p04', 2, 'LogisticRegression 학습')
def p04_q2(v):
    _param(v.lr, 'max_iter', 1000, 'lr')
    need(_fitted(v.lr), 'lr.fit(X_train, y_train) 을 실행하세요.')
    return ''


@_q('p04', 3, '예측 predict')
def p04_q3(v):
    need(np.array_equal(v.lr_pred, v.lr.predict(v.X_test)), 'lr_pred = lr.predict(X_test)')
    return ''


@_q('p04', 4, '분류 지표 4종')
def p04_q4(v):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    yt, yp = v.y_test, v.lr_pred
    need(_close(v.lr_acc, accuracy_score(yt, yp)), 'lr_acc = accuracy_score(y_test, lr_pred)  ← (정답, 예측) 순서!')
    need(_close(v.lr_prec, precision_score(yt, yp)), 'lr_prec = precision_score(y_test, lr_pred)')
    need(_close(v.lr_rec, recall_score(yt, yp)), 'lr_rec = recall_score(y_test, lr_pred)')
    need(_close(v.lr_f1, f1_score(yt, yp)), 'lr_f1 = f1_score(y_test, lr_pred)')
    return f'정확도 {v.lr_acc:.3f} / 재현율 {v.lr_rec:.3f} ← 정확도는 높은데 재현율은 낮죠?'


@_q('p04', 5, '혼동행렬')
def p04_q5(v):
    from sklearn.metrics import confusion_matrix
    need(np.array_equal(v.cm, confusion_matrix(v.y_test, v.lr_pred)), 'cm = confusion_matrix(y_test, lr_pred)')
    return f'[[TN FP] [FN TP]] = {v.cm.tolist()}'


@_q('p04', 6, 'classification_report')
def p04_q6(v):
    need(isinstance(v.report, str) and 'precision' in v.report, 'report = classification_report(y_test, lr_pred)')
    return ''


@_q('p04', 7, 'KNN')
def p04_q7(v):
    _param(v.knn, 'n_neighbors', 5, 'knn')
    need(_fitted(v.knn), 'knn.fit(...)')
    need(len(v.knn_pred) == len(v.y_test), 'knn_pred = knn.predict(X_test)')
    return ''


@_q('p04', 8, 'DecisionTree')
def p04_q8(v):
    _param(v.dt, 'max_depth', 5, 'dt')
    _param(v.dt, 'random_state', 42, 'dt')
    need(_fitted(v.dt) and len(v.dt_pred) == len(v.y_test), 'fit 후 dt_pred = dt.predict(X_test)')
    return ''


@_q('p04', 9, 'RandomForest')
def p04_q9(v):
    for k, val in [('n_estimators', 100), ('max_depth', 8), ('random_state', 42)]:
        _param(v.rf, k, val, 'rf')
    need(_fitted(v.rf) and len(v.rf_pred) == len(v.y_test), 'fit 후 rf_pred = rf.predict(X_test)')
    return ''


@_q('p04', 10, 'XGBoost')
def p04_q10(v):
    for k, val in [('n_estimators', 100), ('max_depth', 4), ('learning_rate', 0.1), ('random_state', 42)]:
        _param(v.xgb, k, val, 'xgb')
    need(len(v.xgb_pred) == len(v.y_test), 'xgb_pred = xgb.predict(X_test)')
    return ''


@_q('p04', 11, 'LightGBM')
def p04_q11(v):
    _param(v.lgbm, 'n_estimators', 100, 'lgbm')
    _param(v.lgbm, 'random_state', 42, 'lgbm')
    need(len(v.lgbm_pred) == len(v.y_test), 'lgbm_pred = lgbm.predict(X_test)')
    return ''


@_q('p04', 12, '최고 모델 고르기')
def p04_q12(v):
    need(v.best_model_name == v.results['f1'].idxmax(), "results['f1'].idxmax() → 가장 큰 값의 인덱스(모델명)")
    return f'F1 1등: {v.best_model_name}'


@_q('p04', 13, '특성 중요도')
def p04_q13(v):
    s = v.importances
    need(isinstance(s, pd.Series) and len(s) == len(v.feature_names), 'pd.Series(rf.feature_importances_, index=feature_names)')
    need(s.is_monotonic_decreasing, 'sort_values(ascending=False) 로 큰 순서대로!')
    return f'Top3: {list(s.index[:3])}'


@_q('p04', 14, '임계값 낮추기 (재현율↑)')
def p04_q14(v):
    exp = (v.rf.predict_proba(v.X_test)[:, 1] >= 0.3).astype(int)
    need(np.array_equal(np.asarray(v.pred_03), exp), 'proba = rf.predict_proba(X_test)[:, 1] → (proba >= 0.3).astype(int)')
    from sklearn.metrics import recall_score
    r0, r1 = recall_score(v.y_test, v.rf_pred), recall_score(v.y_test, exp)
    return f'재현율 {r0:.3f} → {r1:.3f}'


@_q('p04', 15, 'SMOTE 오버샘플링')
def p04_q15(v):
    counts = pd.Series(np.asarray(v.y_over)).value_counts()
    need(len(counts) == 2 and counts.iloc[0] == counts.iloc[1], 'smote.fit_resample(X_train, y_train) → 0과 1의 개수가 같아져요.')
    need(len(v.y_over) > len(v.y_train), '오버샘플링은 학습 데이터에만!')
    return f'{len(v.y_train)} → {len(v.y_over)}행'


@_q('p04', 16, 'LinearRegression')
def p04_q16(v):
    need(_fitted(v.lin), 'lin.fit(Xh_train, yh_train)')
    need(len(v.lin_pred) == len(v.yh_test), 'lin_pred = lin.predict(Xh_test)')
    return ''


@_q('p04', 17, 'DecisionTreeRegressor (샘플문항 파라미터)')
def p04_q17(v):
    for k, val in [('max_depth', 5), ('min_samples_split', 3), ('random_state', 120)]:
        _param(v.dtr, k, val, 'dtr')
    need(_fitted(v.dtr) and len(v.dtr_pred) == len(v.yh_test), 'fit 후 dtr_pred = dtr.predict(Xh_test)')
    return ''


@_q('p04', 18, 'RandomForestRegressor')
def p04_q18(v):
    _param(v.rfr, 'n_estimators', 100, 'rfr')
    _param(v.rfr, 'random_state', 42, 'rfr')
    need(_fitted(v.rfr) and len(v.rfr_pred) == len(v.yh_test), 'fit 후 rfr_pred = rfr.predict(Xh_test)')
    return ''


@_q('p04', 19, '회귀 지표')
def p04_q19(v):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yt, yp = v.yh_test, v.rfr_pred
    need(_close(v.rfr_mae, mean_absolute_error(yt, yp)), 'rfr_mae = mean_absolute_error(yh_test, rfr_pred)')
    need(_close(v.rfr_mse, mean_squared_error(yt, yp)), 'rfr_mse = mean_squared_error(yh_test, rfr_pred)')
    need(_close(v.rfr_rmse, np.sqrt(mean_squared_error(yt, yp))), 'rfr_rmse = np.sqrt(rfr_mse)')
    need(_close(v.rfr_r2, r2_score(yt, yp)), 'rfr_r2 = r2_score(yh_test, rfr_pred)')
    return f'MAE {v.rfr_mae:,.0f}만원, R² {v.rfr_r2:.3f}'


@_q('p04', 20, '최고 회귀 모델')
def p04_q20(v):
    need(v.best_reg_name == v.reg_results['MAE'].idxmin(), "MAE 는 작을수록 좋아요 → idxmin()")
    return f'MAE 1등: {v.best_reg_name}'


# ===========================================================================
# 05 딥러닝 (PyTorch)
# ===========================================================================
def _torch():
    import torch
    return torch


def _layers(model, kind):
    return [m for m in model.modules() if m.__class__.__name__ == kind]


def _linear_dims(model):
    return [(l.in_features, l.out_features) for l in _layers(model, 'Linear')]


def _seq_kinds(model):
    """nn.Sequential 의 층 이름 순서 (예: ['Linear', 'ReLU', ...])."""
    return [m.__class__.__name__ for m in model.children()]


@_q('p05', 1, 'torch import')
def p05_q1(v):
    import torch
    import torch.nn
    from torch.utils.data import DataLoader, TensorDataset
    need(v.torch is torch, 'import torch')
    need(v.nn is torch.nn, 'import torch.nn as nn')
    need(v.TensorDataset is TensorDataset and v.DataLoader is DataLoader,
         'from torch.utils.data import TensorDataset, DataLoader')
    return ''


@_q('p05', 2, 'numpy → 텐서')
def p05_q2(v):
    torch = _torch()
    for name in ['Xtr', 'Xte']:
        t = getattr(v, name)
        need(isinstance(t, torch.Tensor) and t.dtype == torch.float32, f'{name} 는 dtype=torch.float32 텐서')
        need(t.ndim == 2, f'{name} 는 (샘플 수, 특성 수) 2차원')
    for name, X in [('ytr', 'Xtr'), ('yte', 'Xte')]:
        t = getattr(v, name)
        need(isinstance(t, torch.Tensor) and t.dtype == torch.float32, f'{name} 는 dtype=torch.float32 (BCEWithLogitsLoss 정답은 실수)')
        need(tuple(t.shape) == (len(getattr(v, X)), 1), f'{name} 모양은 (N, 1) → .reshape(-1, 1)')
    return f'Xtr {tuple(v.Xtr.shape)}, ytr {tuple(v.ytr.shape)}'


@_q('p05', 3, 'TensorDataset + DataLoader')
def p05_q3(v):
    from torch.utils.data import DataLoader, TensorDataset
    need(isinstance(v.train_ds, TensorDataset) and len(v.train_ds) == len(v.Xtr), 'train_ds = TensorDataset(Xtr, ytr)')
    need(isinstance(v.train_loader, DataLoader) and v.train_loader.batch_size == 32, 'DataLoader(train_ds, batch_size=32, ...)')
    need(v.train_loader.sampler.__class__.__name__ == 'RandomSampler', 'shuffle=True 로 매 epoch 섞어 주세요.')
    return f'배치 수 {len(v.train_loader)}'


@_q('p05', 4, '이진분류 모델 구조')
def p05_q4(v):
    m = v.model
    need(_linear_dims(m) == [(v.n_features, 64), (64, 32), (32, 1)],
         f'Linear 크기: ({v.n_features}→64) → (64→32) → (32→1). 지금: {_linear_dims(m)}')
    drops = _layers(m, 'Dropout')
    need(len(drops) == 1 and _close(drops[0].p, 0.3), 'nn.Dropout(0.3) 하나')
    need(len(_layers(m, 'ReLU')) == 2, 'nn.ReLU() 두 개 (각 은닉층 뒤)')
    need(not _layers(m, 'Sigmoid'), '출력층 뒤에 nn.Sigmoid() 를 넣지 마세요 — BCEWithLogitsLoss 안에 들어 있어요.')
    return ''


@_q('p05', 5, '손실함수 + 옵티마이저')
def p05_q5(v):
    torch = _torch()
    need(isinstance(v.loss_fn, torch.nn.BCEWithLogitsLoss), 'loss_fn = nn.BCEWithLogitsLoss()')
    need(isinstance(v.optimizer, torch.optim.Adam), 'torch.optim.Adam(...)')
    need(_close(v.optimizer.param_groups[0]['lr'], 0.001), 'lr=0.001')
    ids = {id(p) for p in v.model.parameters()}
    need(all(id(p) in ids for p in v.optimizer.param_groups[0]['params']), 'Adam(model.parameters(), ...) — 학습할 대상은 model 의 파라미터')
    return ''


@_q('p05', 6, '한 epoch 학습 함수')
def p05_q6(v):
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset
    fn = v.train_one_epoch
    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y = (X[:, :1] > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(4, 1))
    before = net[0].weight.detach().clone()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = fn(net, DataLoader(TensorDataset(X, y), batch_size=16), torch.nn.BCEWithLogitsLoss(), opt)
    need(isinstance(loss, float), '평균 손실을 float 로 돌려주세요 (loss.item() 사용).')
    need(not torch.equal(before, net[0].weight.detach()),
         '가중치가 안 바뀌었어요. zero_grad → backward → step 순서를 확인하세요.')
    need(net.training, '함수 시작에서 model.train() 을 호출하세요.')
    return f'시험용 미니 데이터 평균 손실 {loss:.3f}'


@_q('p05', 7, '평가 함수')
def p05_q7(v):
    torch = _torch()
    net = torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Dropout(0.5))
    net.train()
    X, y = torch.randn(8, 4), torch.zeros(8, 1)
    out = v.evaluate(net, X, y, torch.nn.BCEWithLogitsLoss())
    need(isinstance(out, float), '.item() 으로 float 을 돌려주세요.')
    need(not net.training, 'model.eval() 을 호출해 Dropout 을 꺼야 해요.')
    return ''


@_q('p05', 8, '학습 루프 + history + 조기 종료')
def p05_q8(v):
    h = v.history
    need(isinstance(h, dict) and {'loss', 'val_loss'} <= set(h), "history = {'loss': [...], 'val_loss': [...]}")
    need(len(h['loss']) == len(h['val_loss']) and 0 < len(h['loss']) <= 30, '최대 30 epoch, 두 리스트 길이가 같아야 해요.')
    need(all(isinstance(x, float) for x in h['val_loss']), '.item() 으로 꺼낸 숫자를 기록하세요.')
    need(v.patience == 5, 'patience = 5')
    need(_close(v.best_val, min(h['val_loss'])), 'best_val 은 기록된 val_loss 중 최솟값이어야 해요.')
    need(isinstance(v.best_state, dict) and len(v.best_state) > 0, 'best_state = copy.deepcopy(model.state_dict())')
    return f"{len(h['loss'])} epoch 학습, best val_loss {v.best_val:.4f}"


@_q('p05', 9, '학습 곡선')
def p05_q9(v):
    ax = _axes(v.ax9, 'ax9')
    need(len(ax.lines) >= 2, 'loss 와 val_loss 두 줄을 그리세요.')
    need(ax.get_legend() is not None, "plt.legend(['loss', 'val_loss'])")
    need(ax.get_title() == 'Loss' and ax.get_xlabel() == 'Epochs', "title 'Loss', xlabel 'Epochs'")
    return ''


@_q('p05', 10, '예측 → 0/1 → 정확도')
def p05_q10(v):
    p = np.asarray(v.proba)
    need(p.ndim == 1 and len(p) == len(v.y_test), 'proba 는 .numpy().flatten() 한 1차원 배열')
    need(p.min() >= 0 and p.max() <= 1, 'torch.sigmoid 를 거쳐야 0~1 확률이 돼요.')
    pr = np.asarray(v.pred)
    need(set(np.unique(pr)) <= {0, 1}, 'pred = (proba > 0.5).astype(int)')
    need(_close(v.acc, float((pr == np.asarray(v.y_test)).mean())), 'acc = accuracy_score(y_test, pred)')
    return f'정확도 {float(v.acc):.3f}'


@_q('p05', 11, '회귀 모델 + 손실')
def p05_q11(v):
    torch = _torch()
    need(_linear_dims(v.reg_model) == [(v.Xh_tr.shape[1], 64), (64, 1)], '(입력 수→64) → ReLU → (64→1)')
    need(isinstance(v.reg_loss_fn, torch.nn.MSELoss), 'reg_loss_fn = nn.MSELoss()')
    need(_close(v.reg_opt.param_groups[0]['lr'], 0.01), 'Adam(..., lr=0.01)')
    return ''


@_q('p05', 12, '회귀 학습 + MAE')
def p05_q12(v):
    h = v.reg_history
    need(len(h['loss']) == 50 and len(h['val_loss']) == 50, '50 epoch 동안 loss / val_loss 를 기록하세요.')
    need(h['val_loss'][-1] < h['val_loss'][0], '검증 손실이 줄어들지 않았어요. reg_loader / reg_loss_fn 을 넘겼는지 확인하세요.')
    need(0 < float(v.reg_mae) < 5, 'reg_mae = mean_absolute_error(yh_test, reg_pred)')
    return f'MAE {float(v.reg_mae):.3f}억'


@_q('p05', 13, '다중분류 텐서 dtype')
def p05_q13(v):
    torch = _torch()
    need(v.Xm_tr.dtype == torch.float32, 'Xm_tr 는 float32')
    need(v.ym_tr.dtype == torch.long and v.ym_tr.ndim == 1, 'ym_tr 는 dtype=torch.long, 모양 (N,) — reshape 하지 않아요.')
    return ''


@_q('p05', 14, '다중분류 모델 + 학습')
def p05_q14(v):
    torch = _torch()
    need(_linear_dims(v.mc_model)[-1][1] == 3, '출력층 nn.Linear(32, 3) — 출력 수 = 클래스 수')
    need(not _layers(v.mc_model, 'Softmax'), 'nn.Softmax 를 넣지 마세요 — CrossEntropyLoss 안에 들어 있어요.')
    need(isinstance(v.mc_loss_fn, torch.nn.CrossEntropyLoss), 'mc_loss_fn = nn.CrossEntropyLoss()')
    need(v.mc_loader.batch_size == 16, 'batch_size=16')
    need(float(v.mc_last) < 0.5, '학습 손실이 충분히 줄지 않았어요. 100 epoch 학습했는지 확인하세요.')
    return ''


@_q('p05', 15, 'argmax 클래스 예측')
def p05_q15(v):
    p = np.asarray(v.mc_pred)
    need(p.ndim == 1 and len(p) == len(v.ym_test), 'scores.argmax(dim=1).numpy() → 1차원')
    need(set(np.unique(p)) <= {0, 1, 2}, '클래스 번호 0/1/2 가 나와야 해요.')
    return f'정확도 {float(v.mc_acc):.3f}'


@_q('p05', 16, '[13번] 지정 구조 모델')
def p05_q16(v):
    torch = _torch()
    m = v.model
    need(isinstance(m, torch.nn.Module), 'model 은 nn.Module (nn.Sequential) 이어야 해요.')
    need(_linear_dims(m) == [(v.Xh_tr.shape[1], 128), (128, 64), (64, 1)], f'Linear 크기: (입력→128) → (128→64) → (64→1). 지금: {_linear_dims(m)}')
    need(_seq_kinds(m) == ['Linear', 'ReLU', 'Dropout', 'Linear', 'ReLU', 'Linear'], '순서: Linear → ReLU → Dropout → Linear → ReLU → Linear')
    need(_close(_layers(m, 'Dropout')[0].p, 0.2), 'Dropout(0.2)')
    need(isinstance(v.loss_fn, torch.nn.MSELoss) and _close(v.optimizer.param_groups[0]['lr'], 0.001), 'loss_fn = nn.MSELoss(), Adam lr=0.001')
    need(v.train_loader.batch_size == 16, 'train_loader batch_size=16')
    return ''


@_q('p05', 17, '[13번] history 기록')
def p05_q17(v):
    h = v.history
    need({'loss', 'val_loss', 'val_mae'} <= set(h), "history 키: 'loss', 'val_loss', 'val_mae'")
    need(len(h['loss']) == len(h['val_loss']) == len(h['val_mae']) == 30, '30 epoch × 세 리스트')
    need(h['val_loss'][-1] < h['val_loss'][0], '검증 손실이 줄지 않았어요. train_one_epoch 로 학습했는지 확인하세요.')
    return f"마지막 val_mae {h['val_mae'][-1]:.3f}억"


@_q('p05', 18, '[14번] 학습 곡선 + 검증 MAE')
def p05_q18(v):
    ax = _axes(v.ax14, 'ax14')
    need(len(ax.lines) >= 2 and ax.get_legend() is not None, "loss / val_loss 두 줄 + legend(['loss', 'val_loss'])")
    need(ax.get_title() == 'Model Loss' and ax.get_xlabel() == 'Epochs' and ax.get_ylabel() == 'Loss',
         "title 'Model Loss', xlabel 'Epochs', ylabel 'Loss' — 시험지 문구 그대로")
    need(not v.model.training, '예측 전에 model.eval() 을 호출하세요 (Dropout 끄기).')
    need(0 < float(v.dnn_mae) < 5, 'dnn_mae = mean_absolute_error(yh_test, dnn_pred)')
    return f'DNN MAE {float(v.dnn_mae):.3f}억'


# ---------------------------------------------------------------------------
# 모의고사 채점 함수 (작업자 A 의 _exam_grader.py) — 같은 모듈에서 쓸 수 있게
# ---------------------------------------------------------------------------
try:
    from _exam_grader import grade_exam1, grade_exam2, grade_exam3
except Exception as _e:                       # 실패 원인이 보이도록 안내 함수로 대체
    _exam_import_error = repr(_e)

    def _exam_unavailable(ns=None):
        print('❌ 모의고사 채점기(_exam_grader.py)를 불러오지 못했어요.')
        print('   notebooks 폴더에 _exam_grader.py 가 있는지, 노트북을 notebooks 폴더에서 실행했는지 확인하세요.')
        print('   원인:', _exam_import_error)

    grade_exam1 = grade_exam2 = grade_exam3 = _exam_unavailable
