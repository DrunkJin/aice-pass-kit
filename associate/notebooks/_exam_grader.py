"""모의고사 자동 채점기

노트북 마지막 셀에서
    import aice_check
    aice_check.grade_exam1(globals())
처럼 호출하면 문항별 ✅/❌ 와 총점을 보여 줍니다.

- 답안 변수가 없거나 계산 중 에러가 나도 해당 문항만 0점 처리하고 계속 채점합니다.
- 앞 문항이 틀려도 뒷 문항은 "여러분이 만든 앞 단계 변수" 기준으로 채점하는 경우가 많습니다.
  (예: df_temp 는 여러분의 df 에서 올바르게 걸렀는지 확인)
- 그래프 자체는 채점하지 않습니다. 그래프 문항은 저장하라고 한 변수/값으로 채점합니다.
"""
import math
import os
import warnings

import numpy as np

PASS_SCORE = 80


# ---------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------
class Fail(Exception):
    """채점 실패 사유를 담는 예외"""


def need(ns, *names):
    """ns 에서 변수를 꺼낸다. 없으면 Fail."""
    out = []
    for name in names:
        if name not in ns:
            raise Fail(f'변수 `{name}` 이(가) 없습니다')
        out.append(ns[name])
    return out[0] if len(out) == 1 else out


def ensure(cond, msg):
    if not cond:
        raise Fail(msg)


def close(a, b, tol=1e-6, rel=1e-4):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def data_dir(ns):
    d = ns.get('DATA_DIR', '../data/')
    return d if d.endswith('/') else d + '/'


def is_module(obj, name):
    return getattr(obj, '__name__', None) == name


def df_like(obj):
    import pandas as pd
    return isinstance(obj, pd.DataFrame)


def same_frame(a, b):
    """값/컬럼/행 수가 같은지 (인덱스는 무시, 컬럼 순서는 무시)"""
    import pandas as pd
    if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)):
        return False
    if a.shape != b.shape or set(a.columns) != set(b.columns):
        return False
    a2 = a[sorted(a.columns)].reset_index(drop=True)
    b2 = b[sorted(b.columns)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a2, b2, check_dtype=False, check_exact=False)
        return True
    except AssertionError:
        return False


def fitted(model):
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def params_match(model, expected):
    p = model.get_params()
    bad = [f'{k}={p.get(k)!r} (요구: {v!r})' for k, v in expected.items() if p.get(k) != v]
    ensure(not bad, '하이퍼파라미터가 다릅니다: ' + ', '.join(bad))


# ---------------------------------------------------------------------
# PyTorch 딥러닝 채점 도구 (torch 는 필요할 때만 import)
# ---------------------------------------------------------------------
def torch_parts(model):
    """model 안의 말단 층을 (Linear (in,out) 목록, Dropout p 목록, 층 이름 목록) 으로"""
    import torch.nn as nn
    ensure(isinstance(model, nn.Module), '`model` 이 torch.nn.Module 이 아닙니다 (nn.Sequential(...) 로 만드세요)')
    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    lin = [(m.in_features, m.out_features) for m in leaves if isinstance(m, nn.Linear)]
    drop = [round(float(m.p), 3) for m in leaves if isinstance(m, nn.Dropout)]
    names = [m.__class__.__name__ for m in leaves]
    return lin, drop, names


def check_torch_model(model, linears, dropouts, n_relu=2, forbid=()):
    lin, drop, names = torch_parts(model)
    ensure(lin == linears, f'Linear 층 (입력, 출력) 구성이 다릅니다\n        현재: {lin}\n        요구: {linears}')
    ensure(drop == dropouts, f'Dropout 비율이 다릅니다 (현재 {drop}, 요구 {dropouts})')
    ensure(names.count('ReLU') >= n_relu, f"ReLU 가 {names.count('ReLU')}개입니다 (요구 {n_relu}개)")
    for f, why in forbid:
        ensure(f not in names, f'모델에 {f} 가 있습니다 → {why}')


def check_tensor(t, name, dtype, shape):
    import torch
    ensure(isinstance(t, torch.Tensor), f'`{name}` 이 torch.Tensor 가 아닙니다 (torch.tensor(...) 로 변환)')
    ensure(t.dtype == dtype, f'`{name}` 의 dtype 이 {t.dtype} 입니다 (요구: {dtype})')
    ensure(tuple(t.shape) == tuple(shape), f'`{name}` 의 모양이 {tuple(t.shape)} 입니다 (요구: {tuple(shape)})')


def check_loader(loader, batch_size):
    from torch.utils.data import DataLoader, RandomSampler
    ensure(isinstance(loader, DataLoader), '`train_loader` 가 DataLoader 가 아닙니다')
    ensure(loader.batch_size == batch_size, f'batch_size 가 {loader.batch_size} 입니다 (요구: {batch_size})')
    ensure(isinstance(loader.sampler, RandomSampler), 'shuffle=True 가 아닙니다')


def check_optim(optimizer, model, lr):
    import torch
    ensure(isinstance(optimizer, torch.optim.Adam), '`optimizer` 가 torch.optim.Adam 이 아닙니다')
    ensure(close(optimizer.param_groups[0]['lr'], lr), f"lr 이 {optimizer.param_groups[0]['lr']} 입니다 (요구: {lr})")
    mine = {id(p) for p in model.parameters()}
    theirs = {id(p) for g in optimizer.param_groups for p in g['params']}
    ensure(mine == theirs, 'optimizer 가 지금 model 의 파라미터를 가리키지 않습니다 (model 을 다시 만들었다면 optimizer 도 다시)')


def check_history(h, epochs=None, max_epochs=None):
    """history = {'loss': [...], 'val_loss': [...]} 형태인지, 숫자 리스트로 바꿔 돌려준다"""
    ensure(isinstance(h, dict), "`history` 는 딕셔너리여야 합니다: {'loss': [...], 'val_loss': [...]}")
    out = {}
    for k in ('loss', 'val_loss'):
        ensure(k in h and len(h[k]) >= 1, f"history['{k}'] 가 없거나 비어 있습니다")
        try:
            out[k] = [float(v) for v in h[k]]
        except (TypeError, ValueError):
            raise Fail(f"history['{k}'] 에는 숫자를 넣으세요 (loss.item())")
        ensure(all(math.isfinite(v) for v in out[k]), f"history['{k}'] 에 nan/inf 가 있습니다 (학습이 발산했어요)")
    n = len(out['loss'])
    ensure(n == len(out['val_loss']), f"loss({n}) 와 val_loss({len(out['val_loss'])}) 길이가 다릅니다 — 매 epoch 둘 다 한 번씩 추가")
    if epochs is not None:
        ensure(n == epochs, f'기록된 epoch 수가 {n} 입니다 (요구: {epochs})')
    if max_epochs is not None:
        ensure(n <= max_epochs, f'기록된 epoch 수가 {n} 입니다 (최대 {max_epochs})')
    return out


def eval_output(model, X):
    """eval 모드 + no_grad 로 출력 계산 (채점용 재계산)"""
    import torch
    model.eval()
    with torch.no_grad():
        return model(X)


class Exam:
    def __init__(self, title):
        self.title = title
        self.items = []

    def q(self, no, points, desc):
        def deco(fn):
            self.items.append((no, points, desc, fn))
            return fn
        return deco

    def grade(self, ns):
        total = sum(p for _, p, _, _ in self.items)
        assert total == 100, f'배점 합이 100이 아닙니다: {total}'
        score = 0
        print(f'📋 {self.title} 채점 결과')
        print('─' * 64)
        for no, points, desc, fn in self.items:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fn(ns)
                score += points
                print(f'✅ {no:>2}번 [{points:>2}/{points:>2}점] {desc}')
            except Fail as e:
                print(f'❌ {no:>2}번 [ 0/{points:>2}점] {desc}\n      → {e}')
            except Exception as e:  # 채점 중 예기치 못한 에러
                print(f'❌ {no:>2}번 [ 0/{points:>2}점] {desc}\n      → 확인 중 에러: {type(e).__name__}: {e}')
        print('─' * 64)
        verdict = '🎉 합격! (80점 이상)' if score >= PASS_SCORE else f'불합격 (합격선 {PASS_SCORE}점, {PASS_SCORE - score}점 부족)'
        print(f'총점: {score} / 100  →  {verdict}')
        return score


# =====================================================================
# 모의고사 1 : 배달 도착시간 예측 (회귀)
# =====================================================================
exam1 = Exam('모의고사 1 · 배달 도착시간 예측 (회귀)')


def _orders_riders(ns):
    import pandas as pd
    d = data_dir(ns)
    return pd.read_json(d + 'delivery_orders.json'), pd.read_csv(d + 'delivery_riders.csv')


@exam1.q(1, 4, 'pandas → pd, numpy → np 로 import')
def _e1q1(ns):
    pd_, np_ = need(ns, 'pd', 'np')
    ensure(is_module(pd_, 'pandas'), '`pd` 가 pandas 가 아닙니다')
    ensure(is_module(np_, 'numpy'), '`np` 가 numpy 가 아닙니다')


@exam1.q(2, 4, 'seaborn → sns, matplotlib.pyplot → plt 로 import')
def _e1q2(ns):
    sns_, plt_ = need(ns, 'sns', 'plt')
    ensure(is_module(sns_, 'seaborn'), '`sns` 가 seaborn 이 아닙니다')
    ensure(is_module(plt_, 'matplotlib.pyplot'), '`plt` 가 matplotlib.pyplot 이 아닙니다')


@exam1.q(3, 7, 'json/csv 읽기 → df_orders, df_riders / rider_id 기준 inner merge → df')
def _e1q3(ns):
    import pandas as pd
    o, r = need(ns, 'df_orders', 'df_riders')
    ro, rr = _orders_riders(ns)
    ensure(df_like(o) and o.shape == ro.shape, f'df_orders 모양이 {getattr(o, "shape", None)} 입니다 (기대 {ro.shape})')
    ensure(df_like(r) and r.shape == rr.shape, f'df_riders 모양이 {getattr(r, "shape", None)} 입니다 (기대 {rr.shape})')
    df = need(ns, 'df')
    merged_cols = set(pd.merge(ro, rr, on='rider_id').columns)
    ensure(df_like(df) and set(df.columns) == merged_cols, 'df 의 컬럼이 두 데이터를 rider_id 로 merge 한 결과와 다릅니다')
    ensure('R0301' not in set(df['rider_id']) and not df['Vehicle'].isna().any(),
           'inner merge 가 아닙니다 (라이더 정보가 없는 주문이 남아 있음 → how="inner")')


@exam1.q(4, 7, "Region 이 '-' 인 행 제거 → df")
def _e1q4(ns):
    import pandas as pd
    df = need(ns, 'df')
    ro, rr = _orders_riders(ns)
    m = pd.merge(ro, rr, on='rider_id')
    exp = m[m['Region'] != '-']
    ensure('-' not in set(df['Region']), "df 에 아직 Region == '-' 행이 있습니다")
    ensure(len(df) == len(exp), f'df 행 수가 {len(df)} 입니다 (기대 {len(exp)})')


@exam1.q(5, 6, 'jointplot(Distance_km vs Delivery_Time) → g')
def _e1q5(ns):
    import seaborn as sns
    g = need(ns, 'g')
    ensure(isinstance(g, sns.JointGrid), '`g` 는 sns.jointplot(...) 의 반환값이어야 합니다')
    xl, yl = g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()
    ensure((xl, yl) == ('Distance_km', 'Delivery_Time'), f'x/y 축이 ({xl}, {yl}) 입니다 (요구: x=Distance_km, y=Delivery_Time)')


@exam1.q(6, 7, 'Speed_Per_Hour 150 이상 이상치 제거 → df_temp')
def _e1q6(ns):
    df, t = need(ns, 'df', 'df_temp')
    ensure(df_like(t), 'df_temp 가 DataFrame 이 아닙니다')
    ensure(t['Speed_Per_Hour'].max() < 150, 'df_temp 에 Speed_Per_Hour >= 150 인 행이 남아 있습니다')
    ensure(len(t) == int((df['Speed_Per_Hour'] < 150).sum()), '150 미만인 행은 모두 남아 있어야 합니다 (조건을 확인하세요)')


@exam1.q(7, 7, '결측치가 있는 행 삭제 → df_na')
def _e1q7(ns):
    t, na = need(ns, 'df_temp', 'df_na')
    ensure(df_like(na), 'df_na 가 DataFrame 이 아닙니다')
    ensure(int(na.isna().sum().sum()) == 0, 'df_na 에 결측치가 남아 있습니다')
    ensure(len(na) == len(t.dropna()), f'df_na 행 수가 {len(na)} 입니다 (기대 {len(t.dropna())})')


DROP1 = ['order_id', 'rider_id', 'Order_Time', 'Arrive_Time']


@exam1.q(8, 7, 'order_id, rider_id, Order_Time, Arrive_Time 컬럼 삭제 → df_del')
def _e1q8(ns):
    na, d = need(ns, 'df_na', 'df_del')
    ensure(df_like(d), 'df_del 이 DataFrame 이 아닙니다')
    left = [c for c in DROP1 if c in d.columns]
    ensure(not left, f'아직 삭제되지 않은 컬럼: {left}')
    ensure(set(d.columns) == set(na.columns) - set(DROP1), '삭제하라는 4개 외의 컬럼은 남겨 두어야 합니다')
    ensure(len(d) == len(na), '행 수가 바뀌었습니다 (axis=1 로 컬럼만 삭제)')


@exam1.q(9, 8, 'object 컬럼 전부 원-핫 인코딩 → df_preset')
def _e1q9(ns):
    import pandas as pd
    d, p = need(ns, 'df_del', 'df_preset')
    ensure(df_like(p), 'df_preset 이 DataFrame 이 아닙니다')
    ensure(len(p.select_dtypes('object').columns) == 0, 'df_preset 에 아직 object 컬럼이 있습니다')
    exp = pd.get_dummies(d, columns=list(d.select_dtypes('object').columns))
    ensure(set(p.columns) == set(exp.columns), f'컬럼 구성이 다릅니다 (컬럼 {p.shape[1]}개, 기대 {exp.shape[1]}개)')


@exam1.q(10, 8, 'X/y 분리 후 train_test_split(test_size=0.2, random_state=42)')
def _e1q10(ns):
    from sklearn.model_selection import train_test_split
    p = need(ns, 'df_preset')
    Xtr, Xva, ytr, yva = need(ns, 'X_train', 'X_valid', 'y_train', 'y_valid')
    ex = train_test_split(p.drop('Delivery_Time', axis=1), p['Delivery_Time'], test_size=0.2, random_state=42)
    ensure(Xtr.shape == ex[0].shape and Xva.shape == ex[1].shape,
           f'X_train {Xtr.shape}, X_valid {Xva.shape} (기대 {ex[0].shape}, {ex[1].shape})')
    ensure('Delivery_Time' not in getattr(Xtr, 'columns', []), 'X 에 타깃(Delivery_Time)이 들어 있습니다')
    ensure(np.allclose(np.asarray(yva, dtype=float), ex[3].to_numpy(dtype=float)),
           'y_valid 값이 다릅니다 (random_state=42, test_size=0.2 확인)')


@exam1.q(11, 8, 'DecisionTreeRegressor → dt, RandomForestRegressor → rf 학습')
def _e1q11(ns):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    dt, rf = need(ns, 'dt', 'rf')
    ensure(isinstance(dt, DecisionTreeRegressor), '`dt` 가 DecisionTreeRegressor 가 아닙니다')
    ensure(isinstance(rf, RandomForestRegressor), '`rf` 가 RandomForestRegressor 가 아닙니다')
    params_match(dt, {'max_depth': 5, 'min_samples_split': 3, 'random_state': 120})
    params_match(rf, {'n_estimators': 100, 'max_depth': 12, 'random_state': 42})
    ensure(fitted(dt) and fitted(rf), '모델을 fit 하지 않았습니다')


@exam1.q(12, 8, '검증 데이터 MAE → dt_mae, rf_mae / RMSE → rf_rmse')
def _e1q12(ns):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    dt, rf, Xva, yva = need(ns, 'dt', 'rf', 'X_valid', 'y_valid')
    dt_mae, rf_mae, rf_rmse = need(ns, 'dt_mae', 'rf_mae', 'rf_rmse')
    p1, p2 = dt.predict(Xva), rf.predict(Xva)
    ensure(close(dt_mae, mean_absolute_error(yva, p1)), 'dt_mae 값이 다릅니다 (mean_absolute_error(y_valid, 예측값))')
    ensure(close(rf_mae, mean_absolute_error(yva, p2)), 'rf_mae 값이 다릅니다')
    ensure(close(rf_rmse, np.sqrt(mean_squared_error(yva, p2))), 'rf_rmse 값이 다릅니다 (MSE 에 루트!)')


@exam1.q(13, 10, 'PyTorch 회귀 DNN 학습 → 텐서, train_loader, model, loss_fn, optimizer, history')
def _e1q13(ns):
    import torch
    import torch.nn as nn
    Xtr, ytr, Xva = need(ns, 'X_train', 'y_train', 'X_valid')
    Xt, Xv, yt, yv = need(ns, 'X_train_t', 'X_valid_t', 'y_train_t', 'y_valid_t')
    check_tensor(Xt, 'X_train_t', torch.float32, Xtr.shape)
    check_tensor(Xv, 'X_valid_t', torch.float32, Xva.shape)
    check_tensor(yt, 'y_train_t', torch.float32, (len(ytr), 1))
    check_tensor(yv, 'y_valid_t', torch.float32, (len(Xva), 1))
    loader, model, loss_fn, opt, h = need(ns, 'train_loader', 'model', 'loss_fn', 'optimizer', 'history')
    check_loader(loader, 32)
    check_torch_model(model, [(Xtr.shape[1], 64), (64, 32), (32, 1)], [0.2])
    ensure(isinstance(loss_fn, nn.MSELoss), '`loss_fn` 이 nn.MSELoss() 가 아닙니다 (회귀)')
    check_optim(opt, model, 0.001)
    hd = check_history(h, epochs=50)
    ensure(hd['loss'][-1] < hd['loss'][0], '학습 손실이 줄지 않았습니다 (zero_grad → backward → step 순서 확인)')


@exam1.q(14, 9, 'eval 모드 예측 → dnn_pred, 검증 MAE → dnn_mae (+ loss 그래프)')
def _e1q14(ns):
    from sklearn.metrics import mean_absolute_error
    model, Xv, yva, pred, v = need(ns, 'model', 'X_valid_t', 'y_valid', 'dnn_pred', 'dnn_mae')
    pred = np.asarray(pred)
    ensure(pred.ndim == 1 and len(pred) == len(yva), f'dnn_pred 모양이 {pred.shape} 입니다 → .numpy().ravel() 로 1차원으로')
    expect = eval_output(model, Xv).numpy().ravel()
    ensure(np.allclose(pred, expect, atol=1e-4), 'dnn_pred 가 eval 모드 예측과 다릅니다 (model.eval() + torch.no_grad() 에서 예측했나요?)')
    ensure(close(v, mean_absolute_error(yva, expect), tol=1e-3, rel=1e-3), 'dnn_mae 값이 다릅니다 (mean_absolute_error(y_valid, dnn_pred))')


def grade_exam1(ns):
    return exam1.grade(ns)


# =====================================================================
# 모의고사 2 : 통신사 고객 이탈 예측 (이진분류)
# =====================================================================
exam2 = Exam('모의고사 2 · 고객 이탈 예측 (이진분류)')


def _telecom(ns):
    import pandas as pd
    return pd.read_csv(data_dir(ns) + 'telecom_customers.csv')


@exam2.q(1, 4, 'pd, np, plt, sns import')
def _e2q1(ns):
    pd_, np_, plt_, sns_ = need(ns, 'pd', 'np', 'plt', 'sns')
    ensure(is_module(pd_, 'pandas') and is_module(np_, 'numpy'), 'pd / np 를 확인하세요')
    ensure(is_module(plt_, 'matplotlib.pyplot') and is_module(sns_, 'seaborn'), 'plt / sns 를 확인하세요')


@exam2.q(2, 5, 'telecom_customers.csv 읽기 → df')
def _e2q2(ns):
    df = need(ns, 'df')
    ensure(df_like(df) and df.shape[0] == len(_telecom(ns)), 'df 행 수가 원본과 다릅니다')


@exam2.q(3, 7, "이탈('Yes') 비율 → churn_rate (소수 셋째 자리 반올림)")
def _e2q3(ns):
    v = need(ns, 'churn_rate')
    raw = _telecom(ns)
    exp = round(float((raw['Churn'] == 'Yes').mean()), 3)
    ensure(close(v, exp, tol=5e-4), f'churn_rate = {v} (기대 {exp})')


@exam2.q(4, 7, "total_fee 의 ' ' 를 0 으로 바꾸고 float 변환 (df 에 반영)")
def _e2q4(ns):
    df = need(ns, 'df')
    raw = _telecom(ns)
    ensure(df['total_fee'].dtype.kind == 'f', f"df['total_fee'] 타입이 {df['total_fee'].dtype} 입니다 (float 이어야 함)")
    exp = raw['total_fee'].replace(' ', '0').astype(float)
    ensure(not df['total_fee'].isna().any(), "total_fee 에 결측이 생겼습니다 (' ' → 0 으로 바꿔야 함)")
    ensure(close(df['total_fee'].sum(), exp.sum(), tol=0.5), 'total_fee 합계가 기대값과 다릅니다')


@exam2.q(5, 7, '결측 50% 이상 컬럼·customer_id 삭제, 수치형 중앙값·범주형 최빈값 채우기 → df1')
def _e2q5(ns):
    df, d1 = need(ns, 'df', 'df1')
    ensure(df_like(d1), 'df1 이 DataFrame 이 아닙니다')
    ensure('promo_code' not in d1.columns, 'promo_code(결측 72%) 컬럼을 삭제해야 합니다')
    ensure('customer_id' not in d1.columns, 'customer_id 컬럼을 삭제해야 합니다')
    ensure(int(d1.isna().sum().sum()) == 0, 'df1 에 결측치가 남아 있습니다')
    ensure(len(d1) == len(df), '행을 삭제하면 안 됩니다 (채우기 문제)')
    miss = df['data_usage_gb'].isna().to_numpy()
    ensure(np.allclose(d1['data_usage_gb'].to_numpy()[miss], df['data_usage_gb'].median()),
           'data_usage_gb 는 중앙값(median)으로 채워야 합니다')
    miss = df['internet'].isna().to_numpy()
    ensure(set(d1['internet'].to_numpy()[miss]) == {df['internet'].mode()[0]},
           'internet 은 최빈값(mode()[0])으로 채워야 합니다')


@exam2.q(6, 7, 'LabelEncoder 로 Churn 인코딩 → le, df1["Churn"]')
def _e2q6(ns):
    from sklearn.preprocessing import LabelEncoder
    le, d1 = need(ns, 'le', 'df1')
    ensure(isinstance(le, LabelEncoder), '`le` 가 LabelEncoder 가 아닙니다')
    ensure(list(le.classes_) == ['No', 'Yes'], f'le.classes_ = {list(getattr(le, "classes_", []))}')
    ensure(set(d1['Churn'].unique()) == {0, 1}, "df1['Churn'] 이 0/1 이 아닙니다")


@exam2.q(7, 7, '이탈 고객(Churn=1)의 tenure_months 중앙값 → churn_tenure_median')
def _e2q7(ns):
    d1, v = need(ns, 'df1', 'churn_tenure_median')
    exp = float(d1.groupby('Churn')['tenure_months'].median()[1])
    ensure(close(v, exp), f'churn_tenure_median = {v} (기대 {exp})')


@exam2.q(8, 7, 'object 컬럼 원-핫 인코딩(drop_first=True) → df_preset')
def _e2q8(ns):
    import pandas as pd
    d1, p = need(ns, 'df1', 'df_preset')
    ensure(len(p.select_dtypes('object').columns) == 0, 'df_preset 에 object 컬럼이 남아 있습니다')
    exp = pd.get_dummies(d1, columns=list(d1.select_dtypes('object').columns), drop_first=True)
    ensure(set(p.columns) == set(exp.columns), f'컬럼 구성이 다릅니다 ({p.shape[1]}개, 기대 {exp.shape[1]}개) — drop_first=True 확인')


@exam2.q(9, 8, 'train_test_split(test_size=0.2, stratify=y, random_state=42)')
def _e2q9(ns):
    from sklearn.model_selection import train_test_split
    p = need(ns, 'df_preset')
    Xtr, Xte, ytr, yte = need(ns, 'X_train', 'X_test', 'y_train', 'y_test')
    X, y = p.drop('Churn', axis=1), p['Churn']
    ex = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ensure(Xtr.shape == ex[0].shape, f'X_train 모양 {Xtr.shape} (기대 {ex[0].shape})')
    ensure(np.array_equal(np.asarray(yte), ex[3].to_numpy()), 'y_test 가 다릅니다 (stratify=y, random_state=42 확인)')


@exam2.q(10, 7, 'StandardScaler: train 으로 fit → X_train_s, X_test_s')
def _e2q10(ns):
    from sklearn.preprocessing import StandardScaler
    sc, Xtr, Xte, a, b = need(ns, 'scaler', 'X_train', 'X_test', 'X_train_s', 'X_test_s')
    ensure(isinstance(sc, StandardScaler), '`scaler` 가 StandardScaler 가 아닙니다')
    ensure(np.allclose(np.asarray(a).mean(axis=0), 0, atol=1e-6), 'X_train_s 의 평균이 0 이 아닙니다 (fit_transform)')
    ref = StandardScaler().fit(Xtr)
    ensure(np.allclose(np.asarray(b), ref.transform(Xte)), 'X_test_s 는 train 으로 fit 한 scaler 로 transform 해야 합니다 (test 에 fit 금지)')


@exam2.q(11, 8, 'RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42) → rf')
def _e2q11(ns):
    from sklearn.ensemble import RandomForestClassifier
    rf = need(ns, 'rf')
    ensure(isinstance(rf, RandomForestClassifier), '`rf` 가 RandomForestClassifier 가 아닙니다')
    params_match(rf, {'n_estimators': 200, 'max_depth': 8, 'random_state': 42})
    ensure(fitted(rf), 'rf 를 fit 하지 않았습니다')


@exam2.q(12, 9, '예측 rf_pred, accuracy → rf_acc, f1 → rf_f1, 혼동행렬 → cm')
def _e2q12(ns):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    rf, Xte, yte = need(ns, 'rf', 'X_test_s', 'y_test')
    pred, acc, f1, cm = need(ns, 'rf_pred', 'rf_acc', 'rf_f1', 'cm')
    ensure(np.array_equal(np.asarray(pred), rf.predict(Xte)), 'rf_pred 는 rf.predict(X_test_s) 결과여야 합니다')
    ensure(close(acc, accuracy_score(yte, pred)), 'rf_acc 값이 다릅니다')
    ensure(close(f1, f1_score(yte, pred)), 'rf_f1 값이 다릅니다')
    ensure(np.array_equal(np.asarray(cm), confusion_matrix(yte, pred)), 'cm 이 confusion_matrix(y_test, rf_pred) 와 다릅니다')


@exam2.q(13, 10, 'PyTorch 이진분류 DNN 학습 (BCEWithLogitsLoss) → model, history')
def _e2q13(ns):
    import torch
    import torch.nn as nn
    Xs, Xts, ytr, yte = need(ns, 'X_train_s', 'X_test_s', 'y_train', 'y_test')
    Xt, Xv, yt, yv = need(ns, 'X_train_t', 'X_test_t', 'y_train_t', 'y_test_t')
    check_tensor(Xt, 'X_train_t', torch.float32, np.shape(Xs))
    check_tensor(Xv, 'X_test_t', torch.float32, np.shape(Xts))
    check_tensor(yt, 'y_train_t', torch.float32, (len(ytr), 1))
    check_tensor(yv, 'y_test_t', torch.float32, (len(yte), 1))
    loader, model, loss_fn, opt, h = need(ns, 'train_loader', 'model', 'loss_fn', 'optimizer', 'history')
    check_loader(loader, 32)
    check_torch_model(model, [(np.shape(Xs)[1], 32), (32, 16), (16, 1)], [0.3],
                      forbid=[('Sigmoid', 'BCEWithLogitsLoss 가 시그모이드를 이미 포함합니다. 출력층의 nn.Sigmoid() 를 빼세요')])
    ensure(isinstance(loss_fn, nn.BCEWithLogitsLoss), '`loss_fn` 이 nn.BCEWithLogitsLoss() 가 아닙니다')
    check_optim(opt, model, 0.001)
    hd = check_history(h, epochs=30)
    ensure(hd['loss'][-1] < hd['loss'][0], '학습 손실이 줄지 않았습니다')


@exam2.q(14, 7, 'sigmoid > 0.5 예측 → dnn_pred (1차원 0/1), accuracy → dnn_acc')
def _e2q14(ns):
    import torch
    from sklearn.metrics import accuracy_score
    model, Xv, yte, pred, acc = need(ns, 'model', 'X_test_t', 'y_test', 'dnn_pred', 'dnn_acc')
    pred = np.asarray(pred)
    ensure(pred.ndim == 1 and len(pred) == len(yte), f'dnn_pred 모양이 {pred.shape} 입니다 → .numpy().ravel() 로 1차원으로')
    ensure(set(np.unique(pred)) <= {0, 1}, 'dnn_pred 는 0/1 이어야 합니다 ((torch.sigmoid(출력) > 0.5).int())')
    expect = (torch.sigmoid(eval_output(model, Xv)) > 0.5).int().numpy().ravel()
    ensure(np.array_equal(pred.astype(int), expect), 'dnn_pred 가 eval 모드 예측과 다릅니다 (model.eval() + torch.no_grad(), 임계값 0.5)')
    ensure(close(acc, accuracy_score(yte, pred)), 'dnn_acc 값이 다릅니다')


def grade_exam2(ns):
    return exam2.grade(ns)


# =====================================================================
# 모의고사 3 : 와인 등급 예측 (다중분류)
# =====================================================================
exam3 = Exam('모의고사 3 · 와인 등급 예측 (다중분류)')


def _wine_raw(ns):
    import pandas as pd
    d = data_dir(ns)
    r = pd.read_csv(d + 'wine_red.csv').assign(type='red')
    w = pd.read_csv(d + 'wine_white.csv').assign(type='white')
    return pd.concat([r, w], ignore_index=True)


@exam3.q(1, 4, 'pd, np, plt, sns import')
def _e3q1(ns):
    pd_, np_, plt_, sns_ = need(ns, 'pd', 'np', 'plt', 'sns')
    ensure(is_module(pd_, 'pandas') and is_module(np_, 'numpy'), 'pd / np 를 확인하세요')
    ensure(is_module(plt_, 'matplotlib.pyplot') and is_module(sns_, 'seaborn'), 'plt / sns 를 확인하세요')


@exam3.q(2, 7, "type 컬럼 추가 후 concat(ignore_index=True) → df")
def _e3q2(ns):
    df = need(ns, 'df')
    raw = _wine_raw(ns)
    ensure(df_like(df) and df.shape == raw.shape, f'df 모양 {getattr(df, "shape", None)} (기대 {raw.shape})')
    ensure(df['type'].value_counts().to_dict() == {'white': 4898, 'red': 1599}, "type 값이 'red' 1599개 / 'white' 4898개가 아닙니다")
    ensure(df.index.is_unique, '인덱스가 중복됩니다 → ignore_index=True')


@exam3.q(3, 6, 'grade 별 alcohol 평균 → alcohol_by_grade, 최고 grade → answer3, white 의 high 비율 → high_ratio_white')
def _e3q3(ns):
    s, a, r = need(ns, 'alcohol_by_grade', 'answer3', 'high_ratio_white')
    raw = _wine_raw(ns)
    exp = raw.groupby('grade')['alcohol'].mean()
    ensure(np.allclose(s.sort_index().to_numpy(dtype=float), exp.sort_index().to_numpy()), 'alcohol_by_grade 값이 다릅니다')
    ensure(a == exp.idxmax(), f'answer3 = {a!r} (평균 alcohol 이 가장 높은 grade 가 아님)')
    er = round(float((raw.loc[raw['type'] == 'white', 'grade'] == 'high').mean()), 3)
    ensure(close(r, er, tol=5e-4), f'high_ratio_white = {r} (기대 {er})')


@exam3.q(4, 6, '수치형 상관행렬 → corr, alcohol 과 |상관계수| 최대 컬럼 → answer4')
def _e3q4(ns):
    c, a = need(ns, 'corr', 'answer4')
    raw = _wine_raw(ns)
    exp = raw.corr(numeric_only=True)
    ensure(df_like(c) and c.shape == exp.shape, f'corr 모양 {getattr(c, "shape", None)} (기대 {exp.shape}) — numeric_only=True')
    s = exp['alcohol'].drop('alcohol').abs()
    ensure(a == s.idxmax(), f'answer4 = {a!r} (절댓값 기준으로 다시 보세요)')


@exam3.q(5, 7, '결측치를 각 컬럼 중앙값으로 채우기 → df_fill')
def _e3q5(ns):
    df, f = need(ns, 'df', 'df_fill')
    ensure(int(f.isna().sum().sum()) == 0, 'df_fill 에 결측치가 남아 있습니다')
    ensure(len(f) == len(df), '행을 삭제하면 안 됩니다')
    for col in ['residual_sugar', 'pH', 'sulphates']:
        m = df[col].isna().to_numpy()
        ensure(np.allclose(f[col].to_numpy()[m], df[col].median()), f'{col} 은 중앙값으로 채워야 합니다')


@exam3.q(6, 8, 'total_sulfur_dioxide IQR 이상치 제거 → df_iqr')
def _e3q6(ns):
    f, q = need(ns, 'df_fill', 'df_iqr')
    s = f['total_sulfur_dioxide']
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    keep = s.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    ensure(len(q) == int(keep.sum()), f'df_iqr 행 수 {len(q)} (기대 {int(keep.sum())}) — 경계값 포함(>=, <=)')


@exam3.q(7, 7, "grade 를 low→0, mid→1, high→2 로 변환, wine_id 삭제 → df_clean")
def _e3q7(ns):
    q, c = need(ns, 'df_iqr', 'df_clean')
    ensure('wine_id' not in c.columns, 'wine_id 를 삭제해야 합니다')
    ensure(len(c) == len(q), '행 수가 df_iqr 과 다릅니다')
    exp = q['grade'].map({'low': 0, 'mid': 1, 'high': 2}).to_numpy() if q['grade'].dtype == object else q['grade'].to_numpy()
    ensure(np.array_equal(c['grade'].to_numpy(), exp), "grade 매핑이 다릅니다 (LabelEncoder 는 알파벳 순서 → high=0 이 돼서 틀림)")


@exam3.q(8, 7, "type 원-핫 인코딩(drop_first=True) → df_preset")
def _e3q8(ns):
    c, p = need(ns, 'df_clean', 'df_preset')
    ensure('type_white' in p.columns and 'type_red' not in p.columns and 'type' not in p.columns,
           "df_preset 에는 type_white 하나만 있어야 합니다 (drop_first=True)")
    ensure(p.shape == (len(c), c.shape[1]), f'df_preset 모양 {p.shape} (기대 {(len(c), c.shape[1])})')


@exam3.q(9, 8, 'train_test_split(test_size=0.25, stratify=y, random_state=2025)')
def _e3q9(ns):
    from sklearn.model_selection import train_test_split
    p = need(ns, 'df_preset')
    Xtr, Xte, ytr, yte = need(ns, 'X_train', 'X_test', 'y_train', 'y_test')
    X, y = p.drop('grade', axis=1), p['grade']
    ex = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2025)
    ensure(Xtr.shape == ex[0].shape, f'X_train 모양 {Xtr.shape} (기대 {ex[0].shape})')
    ensure(np.array_equal(np.asarray(yte), ex[3].to_numpy()), 'y_test 가 다릅니다 (stratify, random_state 확인)')


@exam3.q(10, 7, 'MinMaxScaler: train 으로 fit → X_train_s, X_test_s')
def _e3q10(ns):
    from sklearn.preprocessing import MinMaxScaler
    sc, Xtr, Xte, a, b = need(ns, 'scaler', 'X_train', 'X_test', 'X_train_s', 'X_test_s')
    ensure(isinstance(sc, MinMaxScaler), '`scaler` 가 MinMaxScaler 가 아닙니다')
    a = np.asarray(a, dtype=float)
    ensure(np.isclose(a.min(), 0) and np.isclose(a.max(), 1), 'X_train_s 의 범위가 0~1 이 아닙니다')
    ref = MinMaxScaler().fit(Xtr)
    ensure(np.allclose(np.asarray(b, dtype=float), ref.transform(Xte)), 'X_test_s 는 train 기준 scaler 로 transform 해야 합니다')


@exam3.q(11, 8, 'LogisticRegression → lr, XGBClassifier → xgb 학습')
def _e3q11(ns):
    from sklearn.linear_model import LogisticRegression
    lr, xgb = need(ns, 'lr', 'xgb')
    ensure(isinstance(lr, LogisticRegression), '`lr` 이 LogisticRegression 이 아닙니다')
    ensure(xgb.__class__.__name__ == 'XGBClassifier', '`xgb` 가 XGBClassifier 가 아닙니다')
    params_match(lr, {'max_iter': 1000, 'random_state': 42})
    params_match(xgb, {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42})
    ensure(fitted(lr), 'lr 을 fit 하지 않았습니다')
    ensure(hasattr(xgb, 'classes_'), 'xgb 를 fit 하지 않았습니다')


@exam3.q(12, 8, '정확도 lr_acc, xgb_acc / 더 높은 모델 이름 → best_model_name')
def _e3q12(ns):
    from sklearn.metrics import accuracy_score
    lr, xgb, Xte, yte = need(ns, 'lr', 'xgb', 'X_test_s', 'y_test')
    a1, a2, best = need(ns, 'lr_acc', 'xgb_acc', 'best_model_name')
    e1, e2 = accuracy_score(yte, lr.predict(Xte)), accuracy_score(yte, xgb.predict(Xte))
    ensure(close(a1, e1), f'lr_acc = {a1} (기대 {e1:.4f})')
    ensure(close(a2, e2), f'xgb_acc = {a2} (기대 {e2:.4f})')
    ensure(best == ('xgb' if e2 > e1 else 'lr'), f"best_model_name = {best!r} — 'lr' 또는 'xgb' 중 정확도가 높은 쪽")


@exam3.q(13, 10, 'PyTorch 다중분류 DNN + 조기 종료(patience=10) + best_state 복원 → model, history')
def _e3q13(ns):
    import torch
    import torch.nn as nn
    Xs, Xts, ytr, yte = need(ns, 'X_train_s', 'X_test_s', 'y_train', 'y_test')
    Xt, Xv, yt, yv = need(ns, 'X_train_t', 'X_test_t', 'y_train_t', 'y_test_t')
    check_tensor(Xt, 'X_train_t', torch.float32, np.shape(Xs))
    check_tensor(Xv, 'X_test_t', torch.float32, np.shape(Xts))
    check_tensor(yt, 'y_train_t', torch.long, (len(ytr),))
    check_tensor(yv, 'y_test_t', torch.long, (len(yte),))
    loader, model, loss_fn, opt, h, best = need(ns, 'train_loader', 'model', 'loss_fn', 'optimizer', 'history', 'best_state')
    check_loader(loader, 64)
    check_torch_model(model, [(np.shape(Xs)[1], 64), (64, 32), (32, 3)], [0.2],
                      forbid=[('Softmax', 'CrossEntropyLoss 가 softmax 를 이미 포함합니다. nn.Softmax 를 빼세요')])
    ensure(isinstance(loss_fn, nn.CrossEntropyLoss), '`loss_fn` 이 nn.CrossEntropyLoss() 가 아닙니다')
    check_optim(opt, model, 0.001)
    hd = check_history(h, max_epochs=100)
    vl = hd['val_loss']
    n, best_i = len(vl), int(np.argmin(vl))
    if n < 100:
        ensure(n - 1 - best_i == 10, f'조기 종료 시점이 다릅니다: 최저 val_loss 는 {best_i + 1}번째 epoch, 학습은 {n} epoch 에서 멈춤 '
                                     f'(patience=10 이면 {best_i + 11} epoch 에서 멈춰야 함)')
    ensure(isinstance(best, dict) and len(best) > 0, '`best_state` 는 model.state_dict() 의 복사본(딕셔너리)이어야 합니다')
    ensure(os.path.exists('best_model.pt'), "체크포인트 파일 'best_model.pt' 가 없습니다 (torch.save(best_state, 'best_model.pt'))")
    cur = float(loss_fn(eval_output(model, Xv), yv))
    ensure(close(cur, min(vl), tol=1e-4, rel=1e-4),
           f'현재 model 의 val_loss({cur:.4f}) 가 최저값({min(vl):.4f}) 과 다릅니다 → model.load_state_dict(best_state) 로 복원했나요? (deepcopy 필수)')


@exam3.q(14, 7, 'argmax(dim=1) 로 클래스 예측 → y_pred, 정확도 → dnn_acc')
def _e3q14(ns):
    from sklearn.metrics import accuracy_score
    model, Xv, yte, yp, acc = need(ns, 'model', 'X_test_t', 'y_test', 'y_pred', 'dnn_acc')
    yp = np.asarray(yp)
    ensure(yp.ndim == 1 and len(yp) == len(yte), f'y_pred 모양이 {yp.shape} 입니다 → argmax(dim=1)')
    ensure(set(np.unique(yp)) <= {0, 1, 2}, 'y_pred 는 0/1/2 클래스 번호여야 합니다')
    expect = eval_output(model, Xv).argmax(dim=1).numpy()
    ensure(np.array_equal(yp.astype(int), expect), 'y_pred 가 eval 모드 예측과 다릅니다 (model.eval() + torch.no_grad() + argmax(dim=1))')
    ensure(close(acc, accuracy_score(yte, yp)), 'dnn_acc 값이 다릅니다')


def grade_exam3(ns):
    return exam3.grade(ns)
