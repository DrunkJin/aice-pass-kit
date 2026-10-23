"""모의고사 데이터 생성 스크립트 (재현용)

이 스크립트는 모의고사 1~3에 쓰는 가상 데이터를 만듭니다.
seed 가 고정돼 있어서 몇 번을 실행해도 같은 파일이 나옵니다.
(실습용 데이터는 _generate_practice_data.py 가 따로 만듭니다.)

실행:  python _generate_datasets.py      (data 폴더 안에서)

일부러 넣어 둔 시험식 함정
  - 잘못된 범주값 '-'                       (모의고사1 Region)
  - 측정 오류 이상치                         (모의고사1 Speed_Per_Hour, 모의고사3 total_sulfur_dioxide)
  - 결측치                                   (전 데이터)
  - 공백(' ') 문자열이 섞인 숫자 컬럼         (모의고사2 total_fee)
  - 대부분이 결측인 컬럼                     (모의고사2 promo_code)
  - 삭제해야 할 ID / 시간 문자열 컬럼        (order_id, Order_Time, customer_id, wine_id ...)
  - 클래스 불균형                            (모의고사2 Churn, 모의고사3 grade)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


# =====================================================================
# 모의고사 데이터
# =====================================================================

# ---------------------------------------------------------------------
# 모의고사 1 : 배달 도착시간 예측 (회귀)  → delivery_orders.json + delivery_riders.csv
# ---------------------------------------------------------------------
def make_delivery(seed: int = 7):
    rng = np.random.default_rng(seed)

    # --- 라이더 정보 (csv) ---
    n_riders = 300
    riders = pd.DataFrame({
        'rider_id': [f'R{i:04d}' for i in range(1, n_riders + 1)],
        'Rider_Age': rng.integers(20, 61, n_riders),
        'Vehicle': rng.choice(['Motorcycle', 'Bicycle', 'Car'], n_riders, p=[0.7, 0.15, 0.15]),
        'Experience_Months': rng.integers(1, 97, n_riders),
        'Rider_Rating': np.round(np.clip(rng.normal(4.4, 0.35, n_riders), 3.0, 5.0), 1),
    })
    riders.loc[rng.choice(n_riders, 9, replace=False), 'Rider_Rating'] = np.nan

    # --- 주문 정보 (json) ---
    n = 6000
    # 1% 는 라이더 파일에 없는 라이더(퇴사자) → inner merge 시 빠진다
    known = rng.choice(riders['rider_id'], n)
    unknown = np.array([f'R{i:04d}' for i in rng.integers(301, 321, n)])
    rider_id = np.where(rng.random(n) < 0.01, unknown, known)

    regions = ['Gangnam', 'Seocho', 'Mapo', 'Songpa', 'Yongsan', 'Jongno']
    region = rng.choice(regions, n, p=[0.24, 0.18, 0.17, 0.17, 0.12, 0.12]).astype(object)
    weather = rng.choice(['Sunny', 'Cloudy', 'Rainy', 'Snowy'], n, p=[0.5, 0.25, 0.18, 0.07]).astype(object)
    food = rng.choice(['Chicken', 'Pizza', 'Korean', 'Chinese', 'Cafe', 'Burger'], n,
                      p=[0.25, 0.15, 0.2, 0.15, 0.1, 0.15])
    items = rng.integers(1, 7, n).astype(float)
    hour = rng.choice(np.arange(10, 24), n)
    day = rng.integers(1, 31, n)
    minute = rng.integers(0, 60, n)
    distance = np.round(np.clip(rng.gamma(2.2, 1.3, n), 0.3, 12), 2)

    prep_base = pd.Series(food).map({'Pizza': 18, 'Chicken': 20, 'Korean': 15,
                                     'Chinese': 12, 'Cafe': 7, 'Burger': 10}).to_numpy()
    prep = np.round(np.clip(prep_base + items * 1.5 + rng.normal(0, 3, n), 3, None), 1)

    rmap = riders.set_index('rider_id')
    vehicle = pd.Series(rider_id).map(rmap['Vehicle']).fillna('Motorcycle').to_numpy()
    exp = pd.Series(rider_id).map(rmap['Experience_Months']).fillna(24).to_numpy()
    base_speed = pd.Series(vehicle).map({'Motorcycle': 25, 'Bicycle': 13, 'Car': 20}).to_numpy()
    w_factor = pd.Series(weather).map({'Sunny': 1.0, 'Cloudy': 0.97, 'Rainy': 0.85, 'Snowy': 0.7}).to_numpy()
    rush = np.where(np.isin(hour, [11, 12, 13, 18, 19, 20]), 0.85, 1.0)
    speed = base_speed * w_factor * rush * (1 + exp / 400) * rng.normal(1, 0.08, n)
    travel = distance / speed * 60
    region_add = pd.Series(region).map({'Gangnam': 3, 'Jongno': 2, 'Seocho': 1.5,
                                        'Songpa': 1, 'Mapo': 0.5, 'Yongsan': 0}).to_numpy()
    delivery = np.round(prep + travel + region_add + rng.uniform(2, 5, n), 1)
    speed_obs = np.round(distance / travel * 60, 1)

    order_ts = pd.to_datetime({'year': 2025, 'month': 5, 'day': day, 'hour': hour, 'minute': minute})
    arrive_ts = order_ts + pd.to_timedelta(delivery, unit='m')

    orders = pd.DataFrame({
        'order_id': [f'O{i:06d}' for i in range(1, n + 1)],
        'rider_id': rider_id,
        'Order_Time': order_ts.dt.strftime('%Y-%m-%d %H:%M'),
        'Arrive_Time': arrive_ts.dt.strftime('%Y-%m-%d %H:%M'),
        'Region': region,
        'Weather': weather,
        'Food_Type': food,
        'Items': items,
        'Hour': hour,
        'Distance_km': distance,
        'Prep_Time': prep,
        'Speed_Per_Hour': speed_obs,
        'Delivery_Time': delivery,
    })
    # 함정 1: 지역 미기재 '-' (약 2%)
    orders.loc[rng.choice(n, 120, replace=False), 'Region'] = '-'
    # 함정 2: GPS 오류로 말도 안 되는 시속 (약 1%)
    bad = rng.choice(n, 60, replace=False)
    orders.loc[bad, 'Speed_Per_Hour'] = np.round(rng.uniform(150, 999, 60), 1)
    # 함정 3: 결측치
    orders.loc[rng.choice(n, 110, replace=False), 'Weather'] = np.nan
    orders.loc[rng.choice(n, 70, replace=False), 'Items'] = np.nan

    orders.to_json(OUT / 'delivery_orders.json', orient='records', force_ascii=False, indent=1)
    riders.to_csv(OUT / 'delivery_riders.csv', index=False)
    return orders, riders


# ---------------------------------------------------------------------
# 모의고사 2 : 통신사 고객 이탈 예측 (이진분류) → telecom_customers.csv
# ---------------------------------------------------------------------
def make_telecom(seed: int = 11):
    rng = np.random.default_rng(seed)
    n = 5000
    tenure = np.clip(np.round(rng.exponential(28, n)), 1, 72).astype(int)
    tenure[rng.choice(n, 28, replace=False)] = 0           # 신규 가입자 → total_fee 공백
    contract = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21])
    internet = rng.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]).astype(object)
    streaming = np.where(internet == 'No', 'No', rng.choice(['Yes', 'No'], n, p=[0.45, 0.55]))
    payment = rng.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n,
                         p=[0.34, 0.22, 0.22, 0.22])
    paperless = rng.choice(['Yes', 'No'], n, p=[0.6, 0.4])
    senior = (rng.random(n) < 0.16).astype(int)
    gender = rng.choice(['Male', 'Female'], n)
    partner = rng.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    dependents = np.where(partner == 'Yes', rng.choice(['Yes', 'No'], n, p=[0.5, 0.5]),
                          rng.choice(['Yes', 'No'], n, p=[0.1, 0.9]))
    fee_base = pd.Series(internet).map({'DSL': 55, 'Fiber optic': 82, 'No': 22}).to_numpy()
    monthly = np.round(fee_base + np.where(streaming == 'Yes', 12, 0) + rng.normal(0, 6, n), 2)
    total = np.round(monthly * tenure * rng.normal(1, 0.03, n), 2)
    tickets = rng.poisson(np.where(internet == 'Fiber optic', 1.4, 0.9))
    usage = np.round(np.where(internet == 'No', rng.uniform(0, 2, n),
                              rng.gamma(3, 25, n) * np.where(streaming == 'Yes', 1.6, 1.0)), 1)

    logit = (-2.25
             + pd.Series(contract).map({'Month-to-month': 1.4, 'One year': 0.0, 'Two year': -1.4}).to_numpy()
             + np.where(internet == 'Fiber optic', 0.6, 0)
             + np.where(payment == 'Electronic check', 0.55, 0)
             - 0.045 * tenure
             + 0.42 * tickets
             + 0.35 * senior
             + np.where(paperless == 'Yes', 0.2, 0)
             + 0.012 * (monthly - 65)
             + rng.normal(0, 0.35, n))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 'Yes', 'No')

    df = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(1, n + 1)],
        'gender': gender, 'senior': senior, 'partner': partner, 'dependents': dependents,
        'tenure_months': tenure, 'contract': contract, 'internet': internet, 'streaming': streaming,
        'paperless': paperless, 'payment': payment, 'monthly_fee': monthly,
        'total_fee': total.astype(object), 'num_tickets': tickets, 'data_usage_gb': usage,
        'promo_code': rng.choice(['P10', 'P20', 'P30'], n).astype(object), 'Churn': churn,
    })
    # 함정 1: 신규 가입자의 총요금은 공백 문자열
    df.loc[df['tenure_months'] == 0, 'total_fee'] = ' '
    # 함정 2: 대부분 결측인 컬럼
    df.loc[rng.random(n) < 0.72, 'promo_code'] = np.nan
    # 함정 3: 일반 결측
    df.loc[rng.choice(n, 150, replace=False), 'data_usage_gb'] = np.nan
    df.loc[rng.choice(n, 75, replace=False), 'internet'] = np.nan
    df.to_csv(OUT / 'telecom_customers.csv', index=False)
    return df


# ---------------------------------------------------------------------
# 모의고사 3 : 와인 등급 예측 (다중분류) → wine_red.csv + wine_white.csv
# ---------------------------------------------------------------------
def _wine(rng, n, kind):
    red = kind == 'red'
    alcohol = np.clip(rng.normal(10.4 if red else 10.55, 1.05 if red else 1.2, n), 8.0, 14.9)
    vol = np.clip(rng.normal(0.53 if red else 0.28, 0.17 if red else 0.10, n), 0.08, 1.6)
    citric = np.clip(rng.normal(0.27 if red else 0.33, 0.19 if red else 0.12, n), 0, 1.0)
    sugar = np.clip(rng.lognormal(np.log(2.3) if red else np.log(5.0), 0.35 if red else 0.75, n), 0.6, 45)
    chlorides = np.clip(rng.normal(0.087 if red else 0.046, 0.03 if red else 0.015, n), 0.01, 0.4)
    free_so2 = np.clip(rng.normal(16 if red else 35, 10 if red else 15, n), 1, 120)
    total_so2 = np.clip(free_so2 * (2.8 if red else 3.9) + rng.normal(0, 18 if red else 30, n), 6, 300)
    fixed = np.clip(rng.normal(8.3 if red else 6.85, 1.7 if red else 0.84, n), 3.8, 15.9)
    sulphates = np.clip(rng.normal(0.66 if red else 0.49, 0.15 if red else 0.11, n), 0.22, 2.0)
    density = ((0.9967 if red else 0.9940) - 0.0011 * (alcohol - 10.4)
               + 0.00035 * (sugar - (2.3 if red else 5.0)) + rng.normal(0, 0.0006, n))
    ph = np.clip(rng.normal(3.31 if red else 3.19, 0.15, n) - 0.04 * (fixed - 7.5), 2.7, 4.1)

    latent = (0.95 * (alcohol - 10.4) - 3.0 * (vol - 0.4) + 1.6 * (sulphates - 0.55)
              - 4.0 * (chlorides - 0.06) + 0.5 * (citric - 0.3) + rng.normal(0, 0.75, n))
    grade = np.where(latent < -0.35, 'low', np.where(latent < 1.0, 'mid', 'high'))

    prefix = 'R' if red else 'W'
    df = pd.DataFrame({
        'wine_id': [f'{prefix}{i:04d}' for i in range(1, n + 1)],
        'fixed_acidity': fixed.round(1), 'volatile_acidity': vol.round(3), 'citric_acid': citric.round(2),
        'residual_sugar': sugar.round(1), 'chlorides': chlorides.round(3),
        'free_sulfur_dioxide': free_so2.round(0), 'total_sulfur_dioxide': total_so2.round(0),
        'density': density.round(5), 'pH': ph.round(2), 'sulphates': sulphates.round(2),
        'alcohol': alcohol.round(1), 'grade': grade,
    })
    # 함정 1: 측정 오류 이상치
    k = 12 if red else 25
    df.loc[rng.choice(n, k, replace=False), 'total_sulfur_dioxide'] = rng.integers(400, 470, k).astype(float)
    # 함정 2: 결측치
    df.loc[rng.choice(n, int(n * 0.02), replace=False), 'residual_sugar'] = np.nan
    df.loc[rng.choice(n, int(n * 0.015), replace=False), 'pH'] = np.nan
    df.loc[rng.choice(n, int(n * 0.01), replace=False), 'sulphates'] = np.nan
    return df


def make_wine(seed: int = 23):
    rng = np.random.default_rng(seed)
    red = _wine(rng, 1599, 'red')
    white = _wine(rng, 4898, 'white')
    red.to_csv(OUT / 'wine_red.csv', index=False)
    white.to_csv(OUT / 'wine_white.csv', index=False)
    return red, white


if __name__ == '__main__':
    o, r = make_delivery()
    t = make_telecom()
    wr, ww = make_wine()
    print('delivery_orders.json', o.shape, '| delivery_riders.csv', r.shape)
    print('telecom_customers.csv', t.shape, '| Churn 비율', t['Churn'].value_counts(normalize=True).round(3).to_dict())
    print('wine_red.csv', wr.shape, '| wine_white.csv', ww.shape)
    print('grade 분포', pd.concat([wr, ww])['grade'].value_counts(normalize=True).round(3).to_dict())
