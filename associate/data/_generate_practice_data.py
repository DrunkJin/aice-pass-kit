"""실습용 합성 데이터 생성 스크립트 (재현용, seed 고정).

실행:  python _generate_practice_data.py   (data 폴더 안에서)
생성:  telco_churn.csv, customer_info.json, customer_usage.csv, house_price.csv

일부러 넣어 둔 '함정' (전처리 연습용):
  telco_churn.csv
    - TotalCharges 가 숫자처럼 보이지만 문자열이고, 신규 가입자(tenure=0)는 ' ' (공백 한 칸)
    - DeviceProtection 컬럼은 절반 가까이가 결측  → 컬럼째 삭제 대상
    - gender / SeniorCitizen / Dependents / PhoneService / Contract 에 결측 몇 개 → 행 삭제 대상
    - MonthlyCharges 에 입력 오류성 이상치(900 이상) 몇 개
    - Churn 은 Yes 가 약 26% 인 불균형 데이터
  customer_info.json / customer_usage.csv
    - customerID 로 병합(merge) 연습. 양쪽에만 있는 ID 가 있어서 inner / left 결과가 다름
  house_price.csv
    - area_m2 이상치(입력 오류), built_year / station_dist_m 결측, region 은 문자열(범주형)
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2024)


def make_telco(n=3000):
    ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)]
    gender = rng.choice(['Male', 'Female'], n)
    senior = rng.choice([0.0, 1.0], n, p=[0.84, 0.16])
    partner = rng.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    dependents = np.where(partner == 'Yes', rng.choice(['Yes', 'No'], n, p=[0.55, 0.45]),
                          rng.choice(['Yes', 'No'], n, p=[0.1, 0.9]))
    contract = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
    base_tenure = np.select([contract == 'Month-to-month', contract == 'One year'],
                            [rng.integers(0, 40, n), rng.integers(5, 65, n)], rng.integers(15, 73, n))
    tenure = np.clip(base_tenure, 0, 72)
    tenure[rng.choice(n, 12, replace=False)] = 0          # 신규 가입자 → TotalCharges 공백
    phone = rng.choice(['Yes', 'No'], n, p=[0.9, 0.1])
    multiple = np.where(phone == 'No', 'No phone service', rng.choice(['Yes', 'No'], n, p=[0.45, 0.55]))
    internet = rng.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])

    def addon(p_yes):
        v = rng.choice(['Yes', 'No'], n, p=[p_yes, 1 - p_yes])
        return np.where(internet == 'No', 'No internet service', v)

    online_sec = addon(0.35)
    tech = addon(0.35)
    device = addon(0.4)
    paperless = rng.choice(['Yes', 'No'], n, p=[0.6, 0.4])
    payment = rng.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                         n, p=[0.34, 0.23, 0.22, 0.21])
    monthly = (20 + (phone == 'Yes') * 10 + (internet == 'DSL') * 25 + (internet == 'Fiber optic') * 50
               + (online_sec == 'Yes') * 5 + (tech == 'Yes') * 5 + rng.normal(0, 6, n))
    monthly = np.round(np.clip(monthly, 18.25, 118.75), 2)
    total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)

    # 이탈 확률: 월 단위 계약, 짧은 가입기간, 광랜, 기술지원 없음, 전자수표 → 이탈↑
    logit = (-1.6 + 1.5 * (contract == 'Month-to-month') - 1.2 * (contract == 'Two year')
             - 0.035 * tenure + 0.7 * (internet == 'Fiber optic') - 0.5 * (tech == 'Yes')
             + 0.45 * (payment == 'Electronic check') + 0.3 * senior + rng.normal(0, 0.4, n))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 'Yes', 'No')

    df = pd.DataFrame({
        'customerID': ids, 'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple,
        'InternetService': internet, 'OnlineSecurity': online_sec, 'TechSupport': tech,
        'DeviceProtection': device, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly,
        'TotalCharges': [f'{v:.2f}'.rstrip('0').rstrip('.') for v in total], 'Churn': churn,
    })
    df.loc[df['tenure'] == 0, 'TotalCharges'] = ' '
    # 이상치 (입력 오류)
    out_idx = rng.choice(n, 7, replace=False)
    df.loc[out_idx, 'MonthlyCharges'] = rng.uniform(900, 1500, 7).round(2)
    # 결측
    df.loc[rng.choice(n, int(n * 0.48), replace=False), 'DeviceProtection'] = np.nan
    for col, k in [('gender', 9), ('SeniorCitizen', 1), ('Dependents', 2), ('PhoneService', 3), ('Contract', 1)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    df.to_csv(os.path.join(HERE, 'telco_churn.csv'), index=False)
    return df


def make_merge_pair(telco):
    ids = telco['customerID'].tolist()[:520]
    info_ids = ids[:500]                       # info: 0~499
    usage_ids = ids[20:520]                    # usage: 20~519  → 공통 480명
    info = [{'customerID': c,
             'age': int(rng.integers(19, 80)),
             'region': str(rng.choice(['Seoul', 'Busan', 'Incheon', 'Daegu', 'Gwangju', 'Daejeon'])),
             'join_channel': str(rng.choice(['online', 'store', 'phone']))} for c in info_ids]
    with open(os.path.join(HERE, 'customer_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    usage = pd.DataFrame({
        'customerID': usage_ids,
        'data_gb': rng.gamma(2.0, 6.0, len(usage_ids)).round(1),
        'call_min': rng.integers(0, 900, len(usage_ids)),
        'plan': rng.choice(['basic', 'standard', 'premium'], len(usage_ids), p=[0.5, 0.35, 0.15]),
    })
    usage.to_csv(os.path.join(HERE, 'customer_usage.csv'), index=False)


def make_house(n=2000):
    region = rng.choice(['Gangnam', 'Songpa', 'Mapo', 'Seongdong', 'Nowon', 'Guro', 'Gangseo'], n,
                        p=[0.14, 0.15, 0.14, 0.12, 0.17, 0.14, 0.14])
    region_premium = pd.Series(region).map({'Gangnam': 2.1, 'Songpa': 1.7, 'Seongdong': 1.55, 'Mapo': 1.5,
                                            'Gangseo': 1.05, 'Guro': 0.95, 'Nowon': 0.85}).to_numpy()
    area = np.round(rng.normal(84, 22, n).clip(30, 180), 1)
    rooms = np.clip(np.round(area / 30 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    built = rng.integers(1988, 2025, n).astype(float)
    station = np.round(rng.gamma(2.2, 280, n)).clip(50, 3000)
    floor = rng.integers(1, 36, n)
    parking = rng.choice(['Yes', 'No'], n, p=[0.8, 0.2])
    price = (area * 95 * region_premium * (1 + (built - 1988) * 0.008) * (1 - station / 9000)
             * (1 + floor * 0.003) * np.where(parking == 'Yes', 1.04, 1.0) * rng.normal(1, 0.07, n))
    df = pd.DataFrame({'area_m2': area, 'rooms': rooms, 'region': region, 'built_year': built,
                       'station_dist_m': station, 'floor': floor, 'parking': parking,
                       'price': np.round(price, -1)})
    out_idx = rng.choice(n, 6, replace=False)
    df.loc[out_idx, 'area_m2'] = rng.uniform(800, 1500, 6).round(1)   # 입력 오류 (단위 착오)
    df.loc[rng.choice(n, 40, replace=False), 'built_year'] = np.nan
    df.loc[rng.choice(n, 25, replace=False), 'station_dist_m'] = np.nan
    df.to_csv(os.path.join(HERE, 'house_price.csv'), index=False)


if __name__ == '__main__':
    t = make_telco()
    make_merge_pair(t)
    make_house()
    print('done:', sorted(f for f in os.listdir(HERE) if not f.startswith('_')))
