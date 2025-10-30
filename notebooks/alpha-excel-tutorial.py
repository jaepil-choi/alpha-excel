# %% [markdown]
# # Alpha-Excel 시연

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.constants import *
from alpha_excel.ops.arithmetic import *
from alpha_excel.ops.timeseries import *
from alpha_excel.ops.group import *
from alpha_excel.ops.crosssection import *
from alpha_excel.ops.transformation import *
from alpha_excel.portfolio import *

from alpha_database import DataSource

# %% [markdown]
# ## 시작하기

# %%
# DataSource 인스턴스 생성
ds = DataSource()

# AlphaExcel 인스턴스 생성
ae = AlphaExcel(
    data_source=ds,
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# %%
ae.universe.head()

# %%
ae.universe.shape

# %% [markdown]
# ## 데이터 불러오기

# %%
returns = Field("returns") # 수익률
returns

# %%
industry_group = Field("fnguide_industry_group") # 에프앤가이드 산업분류
industry_group

# %% [markdown]
# 데이터를 불러오려면 Expression을 Evaluate 해야 함.

# %%
returns_df = ae.evaluate(returns)
returns_df.head()

# %%
industry_group_df = ae.evaluate(industry_group)
industry_group_df.iloc[18:21] # 산업 분류는 월말에만 데이터가 존재하지만, 불러올 때 자동으로 ffill 처리됨. 

# %% [markdown]
# ## 연산자 사용하기

# %% [markdown]
# ### 코스피 인덱스

# %%
expr = Field('fnguide_market_cap')
scaler = LongOnlyScaler()

result = ae.evaluate(expr, scaler=scaler)

# %%
step0 = ae.get_cumulative_pnl(0)

step0.plot()

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### 롱숏 포트폴리오

# %%
expr = GroupNeutralize(
    TsMean(
        Field('returns') * -1, 
        window=5
        ), 
        group_by='fnguide_industry_group'
    )
expr

# %%
# expr = Field('returns')
# expr

# %%
scaler = DollarNeutralScaler() # Long Short의 합은 0, 절대값 합은 2가 되도록 스케일링
# scaler = LongOnlyScaler() # Long Short의 합은 0, 절대값 합은 1이 되도록 스케일링

# %%
result = ae.evaluate(expr, scaler=scaler)
result.head()

# %%
step0 = ae.get_cumulative_pnl(0) # 'returns' 자체를 시그널로 사용했을 때의 수익률
# step1 = ae.get_cumulative_pnl(1) # -1 곱했을 때의 수익률
# step2 = ae.get_cumulative_pnl(2) # TsMean 씌웠을 때의 수익률
# step3 = ae.get_cumulative_pnl(3) # GroupNeutralize 씌웠을 때의 수익률

# %%
step0

# %%
step0.plot()

# %%
returns = ae.returns
pure_return_signal_port_weights = ae.get_weights(0)
positive_weights = pure_return_signal_port_weights[pure_return_signal_port_weights > 0]
negative_weights = pure_return_signal_port_weights[pure_return_signal_port_weights < 0]

# %%
positive_port_ret = positive_weights.shift(1) * returns
negative_port_ret = negative_weights.shift(1) * returns

# %%
positive_port_pnl = positive_port_ret.sum(axis=1).cumsum()
negative_port_pnl = negative_port_ret.sum(axis=1).cumsum()

# %%
positive_port_pnl.plot()

# %%
negative_port_pnl.plot()

# %%
step0.plot(label='Original Returns')
step1.plot(label='Negated Returns')
step2.plot(label='TsMean Signal')
step3.plot(label='GroupNeutralized Signal')

plt.title('Cumulative PnL of signal evolution')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.legend()
plt.show()


# %% [markdown]
# ### 파마 프랜치 사이즈 팩터
# 

# %%
# Step 1: 독립적인 사이즈 그룹 생성 (2개 bins: Small, Big)
size_group = LabelQuantile(Field('fnguide_market_cap'), bins=2, labels=['Small', 'Big'])
size_groups_df = ae.evaluate(size_group)
size_groups_df.head()

# %%
# 사이즈 그룹 분포 확인
sample_date = size_groups_df.index[30]
print(f"사이즈 분포 ({sample_date.strftime('%Y-%m-%d')}):")
print(size_groups_df.loc[sample_date].value_counts())

# %%
# Step 2: 가치 그룹 생성 (3개 bins: Low, Med, High)
# 주의: 실제 book-to-market 데이터가 없어서 float_ratio를 proxy로 사용
value_group = LabelQuantile(Field('fnguide_float_ratio_pct'), bins=3, labels=['Low', 'Med', 'High'])
value_groups_df = ae.evaluate(value_group)
value_groups_df.head()

# %%
# 가치 그룹 분포 확인
print(f"가치 분포 ({sample_date.strftime('%Y-%m-%d')}):")
print(value_groups_df.loc[sample_date].value_counts())

# %% [markdown]
# ### 복합 그룹 생성 (2×3 = 6 포트폴리오)

# %%
# Step 3: CompositeGroup을 사용하여 사이즈와 가치 그룹을 결합
# 결과: Small&Low, Small&Med, Small&High, Big&Low, Big&Med, Big&High (6개 포트폴리오)
composite_group = CompositeGroup(
    size_group,
    value_group,
    separator='&'
)
composite_groups_df = ae.evaluate(composite_group)
composite_groups_df.head()

# %%
# 포트폴리오 분포 확인
print(f"포트폴리오 분포 ({sample_date.strftime('%Y-%m-%d')}):")
print(composite_groups_df.loc[sample_date].value_counts().sort_index())

# %% [markdown]
# ### SMB 팩터 시그널 할당

# %%
mask_g1 = np.where(size > np.nanpercentile(size, 2/3), True, False)


# %%
# Step 4: SMB 팩터를 위한 방향성 시그널 할당
# SMB = Small 포트폴리오 평균 - Big 포트폴리오 평균
# Small 포트폴리오: +1/3 (3개 포트폴리오 × 1/3 = +1.0)
# Big 포트폴리오: -1/3 (3개 포트폴리오 × -1/3 = -1.0)

smb_mapping = {
    'Small&Low': 1/3, 'Small&Med': 1/3, 'Small&High': 1/3,
    'Big&Low': -1/3, 'Big&Med': -1/3, 'Big&High': -1/3
}

smb_signals = MapValues(composite_group, mapping=smb_mapping)
smb_signals_df = ae.evaluate(smb_signals)
smb_signals_df.head()

# %%
# 시그널 분포 및 Net Exposure 확인
print(f"시그널 분포 ({sample_date.strftime('%Y-%m-%d')}):")
print(smb_signals_df.loc[sample_date].value_counts().sort_index())
print(f"\nNet Exposure: {smb_signals_df.loc[sample_date].sum():.6f} (목표: 0.0)")

# %% [markdown]
# ### 포트폴리오 내 시가총액 가중치 적용

# %%
# Step 5: GroupScalePositive를 사용하여 각 포트폴리오 내에서 시가총액 가중치 적용
# 각 주식의 가중치 = (시가총액 / 포트폴리오 내 시가총액 합계)
# 각 포트폴리오는 독립적으로 1.0으로 정규화됨

value_weights = GroupScalePositive(
    Field('fnguide_market_cap'),
    group_by=composite_group
)
value_weights_df = ae.evaluate(value_weights)
value_weights_df.head()

# %%
# 포트폴리오별 가중치 합계 확인 (각 포트폴리오는 1.0이 되어야 함)
print(f"포트폴리오별 가중치 합계 ({sample_date.strftime('%Y-%m-%d')}):")
for portfolio in composite_groups_df.loc[sample_date].dropna().unique():
    mask = composite_groups_df.loc[sample_date] == portfolio
    portfolio_sum = value_weights_df.loc[sample_date][mask].sum()
    print(f"  {portfolio:15s}: {portfolio_sum:.6f}")

# %% [markdown]
# ### 최종 SMB 포트폴리오 가중치 계산

# %%
# Step 6: 시그널과 가중치를 곱하여 최종 SMB 포트폴리오 가중치 생성
# 최종 가중치 = signal × value_weight
# 각 주식의 가중치 = (±1/3) × (시가총액 / 포트폴리오 내 시가총액 합계)

smb_weights = Multiply(smb_signals, value_weights)
smb_weights_df = ae.evaluate(smb_weights)
smb_weights_df.head()

# %%
# 포트폴리오 익스포저 확인
long_exposure = smb_weights_df.loc[sample_date].where(smb_weights_df.loc[sample_date] > 0, 0).sum()
short_exposure = smb_weights_df.loc[sample_date].where(smb_weights_df.loc[sample_date] < 0, 0).sum()
gross_exposure = smb_weights_df.loc[sample_date].abs().sum()
net_exposure = smb_weights_df.loc[sample_date].sum()

print(f"포트폴리오 익스포저 ({sample_date.strftime('%Y-%m-%d')}):")
print(f"  Long:  {long_exposure:+.6f} (목표: +1.0)")
print(f"  Short: {short_exposure:+.6f} (목표: -1.0)")
print(f"  Gross: {gross_exposure:+.6f} (목표: 2.0)")
print(f"  Net:   {net_exposure:+.6f} (목표: 0.0)")

# %%
# 샘플 가중치 확인 (특정 날짜의 일부 주식들)
market_cap_df = ae.evaluate(Field('fnguide_market_cap'))

sample_weights_df = pd.DataFrame({
    'Market Cap': market_cap_df.loc[sample_date][:10],
    'Portfolio': composite_groups_df.loc[sample_date][:10],
    'Signal': smb_signals_df.loc[sample_date][:10],
    'Value Weight': value_weights_df.loc[sample_date][:10],
    'Final Weight': smb_weights_df.loc[sample_date][:10]
})
sample_weights_df

# %% [markdown]
# ### SMB 팩터 백테스트

# %%
# AlphaExcel의 백테스트 기능을 사용하여 SMB 팩터 성과 평가
# smb_weights를 시그널로 사용하고, DollarNeutralScaler를 적용

scaler = DollarNeutralScaler()
result = ae.evaluate(smb_weights, scaler=scaler)

# 누적 수익률 계산
smb_pnl = ae.get_final_cumulative_pnl()
smb_pnl.head()

# %%
# SMB 팩터 누적 수익률 시각화
plt.figure(figsize=(12, 6))
smb_pnl.plot(linewidth=2)
plt.title('SMB Factor Cumulative PnL (Small Minus Big)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative PnL', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# 통계 요약
print("SMB 팩터 성과 통계:")
print(f"  총 수익률: {smb_pnl.iloc[-1]:.4f}")
print(f"  연율화 수익률: {(smb_pnl.iloc[-1] / len(smb_pnl) * 252):.4f}")
print(f"  샤프 비율 (추정): {(smb_pnl.pct_change().mean() / smb_pnl.pct_change().std() * np.sqrt(252)):.4f}")

# %% [markdown]
# ### 동등가중 SMB 팩터 (비교)

# %%
# 시가총액 대신 Constant(1)을 사용하여 동등가중 포트폴리오 생성
equal_weights = GroupScalePositive(
    Constant(1),
    group_by=composite_group
)
equal_weights_df = ae.evaluate(equal_weights)

# 동등가중 SMB 가중치
smb_weights_ew = Multiply(smb_signals, equal_weights)
smb_weights_ew_df = ae.evaluate(smb_weights_ew)

# %%
# 동등가중 익스포저 확인
long_ew = smb_weights_ew_df.loc[sample_date].where(smb_weights_ew_df.loc[sample_date] > 0, 0).sum()
short_ew = smb_weights_ew_df.loc[sample_date].where(smb_weights_ew_df.loc[sample_date] < 0, 0).sum()

print(f"동등가중 포트폴리오 익스포저 ({sample_date.strftime('%Y-%m-%d')}):")
print(f"  Long:  {long_ew:+.6f}")
print(f"  Short: {short_ew:+.6f}")
print(f"  Net:   {long_ew + short_ew:+.6f}")

# %% [markdown]
# ### 시가총액 가중 vs 동등가중 비교

# %%
# 특정 포트폴리오의 가중치 집중도 비교
sample_portfolio = 'Small&Low'
mask = composite_groups_df.loc[sample_date] == sample_portfolio

if mask.sum() > 0:
    comparison_df = pd.DataFrame({
        'Market Cap': market_cap_df.loc[sample_date][mask],
        'Value Weight': value_weights_df.loc[sample_date][mask],
        'Equal Weight': equal_weights_df.loc[sample_date][mask]
    }).sort_values('Market Cap', ascending=False)
    
    print(f"{sample_portfolio} 포트폴리오 (n={mask.sum()} 주식):")
    print("\n상위 5개 주식 (시가총액 기준):")
    print(comparison_df.head())
    
    print(f"\n가중치 집중도 (상위 5개 주식):")
    print(f"  시가총액 가중: {comparison_df['Value Weight'].head().sum():.1%}")
    print(f"  동등가중: {comparison_df['Equal Weight'].head().sum():.1%}")

# %% [markdown]
# ## 요약
# 
# 이 노트북에서는 AlphaExcel의 주요 기능들을 시연했습니다:
# 
# 1. **기본 데이터 로딩**: `Field`를 사용한 데이터 접근
# 2. **시계열 연산자**: `TsMean` 등을 활용한 시계열 신호 생성
# 3. **그룹 연산자**: `GroupNeutralize`를 통한 산업 중립화
# 4. **파마-프렌치 팩터 구축**:
#    - `LabelQuantile`: 독립적인 사이즈/가치 그룹화
#    - `CompositeGroup`: 2×3 포트폴리오 생성
#    - `MapValues`: 방향성 시그널 할당
#    - `GroupScalePositive`: 포트폴리오 내 가중치 계산
#    - `Multiply`: 시그널과 가중치 결합
# 5. **백테스팅**: 누적 수익률 계산 및 시각화
# 
# ### 핵심 개념
# 
# - **Expression 기반 설계**: 모든 연산은 Expression 객체로 표현되며, `evaluate()`로 실행
# - **지연 평가**: 데이터는 필요할 때만 로드됨
# - **조합 가능성**: 연산자들을 자유롭게 조합하여 복잡한 신호 생성 가능
# - **그룹 연산**: 포트폴리오 구축과 팩터 생성을 위한 강력한 그룹 연산 지원


