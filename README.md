# Alpha Excel

**엑셀처럼 간편한 퀀트 리서치 플랫폼**

Alpha Excel은 pandas 기반의 퀀트 분석 도구 모음으로, 시그널 기반 트레이딩 전략을 엑셀 함수처럼 직관적이고 빠르게 구축할 수 있습니다.

## ✨ 주요 특징

- 🎯 **엑셀 함수 스타일 API**: `o.ts_mean()`, `o.rank()` 등 친숙한 함수 방식으로 시그널 생성
- ⚙️ **설정 기반 데이터 로딩**: `f('market_cap')` 만으로 데이터베이스에서 데이터를 자동 추출 및 전처리하여 제공
- 💾 **선택적 캐싱**: 중요한 중간 결과를 저장하여 전략 성과 개선 과정 추적
- 📊 **타입 인식 시스템**: 데이터 타입(numeric, group, weight)별 자동 전처리
- 🎭 **유니버스 자동 관리**: 투자 대상 유니버스를 모든 연산에서 일관되게 유지
- 💡 **Import 불필요**: 모든 연산자가 메서드로 제공되어 IDE 자동완성 지원
- 🚀 **즉시 실행**: 연산 호출 시 바로 결과 확인 가능, 빠른 프로토타이핑 지원

## 🏗️ 프로젝트 구조

이 저장소는 모노레포 구조로 여러 퀀트 분석 도구를 포함합니다:

- **alpha-excel**: 시그널 생성 및 백테스팅 엔진 (메인 패키지)
- **alpha-database**: 설정 기반 데이터 조회 (Parquet + DuckDB)
- **alpha-lab**: 분석 및 시각화 도구 (개발 예정)

## 📦 설치

현재 개발 중이며 PyPI 배포 전입니다. Git 저장소에서 직접 설치하세요:

```bash
# 저장소 클론
git clone https://github.com/your-org/alpha-excel.git
cd alpha-excel

# Poetry로 설치
poetry install

# 활성화
poetry shell
```

> **참고**: 필수 데이터가 없으면 기능이 제한됩니다. `alpha-database`의 데이터 설정(`config/data.yaml`)을 참고하세요.

## 🚀 빠른 시작

```python
from alpha_excel2.core.facade import AlphaExcel

# 초기화
ae = AlphaExcel(start_time='2023-01-01', end_time='2023-12-31')
f = ae.field  # 데이터 로더
o = ae.ops    # 연산자

# 데이터 로딩 (config/data.yaml에서 자동 조회)
returns = f('returns')
sector = f('fnguide_sector')

# 시그널 생성 (즉시 실행, 결과 바로 확인)
ma5 = o.ts_mean(returns, window=5)
ma20 = o.ts_mean(returns, window=20)
momentum = ma5 - ma20  # 산술 연산 지원

# 순위화 및 결합
signal = o.rank(momentum)
sector_signal = o.group_rank(returns, sector)  # 섹터 내 순위
combined = 0.6 * signal + 0.4 * sector_signal

# 백테스팅
ae.set_scaler('DollarNeutral')
weights = ae.to_weights(combined)
port_returns = ae.to_portfolio_returns(weights)

# 성과 분석
daily_pnl = port_returns.to_df().sum(axis=1)
sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe:.2f}")
```

## 💡 핵심 개념

### 1. 엑셀 함수 스타일 API

모든 데이터 로딩과 연산자가 간단한 함수 호출로 제공되어 import 없이 사용 가능합니다.

```python
f = ae.field
o = ae.ops

# 데이터 로딩 (config/data.yaml 설정 기반)
returns = f('returns')
market_cap = f('market_cap')
industry = f('fnguide_industry_group')

# 연산자 사용
signal0 = o.ts_mean(returns, window=5)      # 시계열 이동평균
signal1 = o.rank(signal)                    # 크로스섹션 순위
signal2 = o.group_rank(signal1, industry)      # 그룹 내 순위

# 사용 가능한 연산자 확인
print(o.list_operators())
```

### 2. 선택적 캐싱

중요한 중간 결과를 `record_output=True`로 저장하여, 전략 개선 과정을 추적하고 분석할 수 있습니다.

```python
# 기본 시그널 (캐싱)
base_signal = o.rank(momentum, record_output=True)

# 개선된 시그널
improved_signal = 0.6 * base_signal + 0.4 * sector_signal

# 나중에 기본 시그널 성과와 비교 가능
cached_base = improved_signal.get_cached_step(base_signal._step_counter)
```

### 3. 타입 인식 시스템

데이터 타입에 따라 자동으로 적절한 전처리가 적용됩니다.

```python
returns = f('returns')       # numeric: 그대로 로딩
sector = f('fnguide_sector') # group: forward-fill 적용, category 변환
```

### 4. 유니버스 자동 관리

투자 대상 유니버스가 모든 필드 로딩과 연산에 자동으로 적용되어 일관성을 유지합니다.

```python
# 초기화 시 유니버스 지정
ae = AlphaExcel(
    start_time='2023-01-01',
    end_time='2023-12-31',
    universe=my_universe_mask  # 모든 연산에 자동 적용
)

# 이후 모든 데이터와 연산 결과가 자동으로 유니버스 필터링됨
```

### 5. 즉시 실행 (Eager Execution)

각 연산이 호출될 때 바로 실행되어 중간 결과를 즉시 확인할 수 있습니다.

```python
ma5 = o.ts_mean(returns, window=5)
print(ma5.to_df().head())  # 바로 DataFrame 확인 가능
```

## 📚 문서 및 예제

- **튜토리얼**: `notebooks/alpha-excel-2-tutorial.ipynb` - 단계별 가이드
- **Showcase**: `showcase/ae2_01_basic_workflow.py` - 전체 워크플로우 데모
- **PRD**: `docs/vibe_coding/alpha-excel/ae2-prd.md` - 제품 요구사항
- **Architecture**: `docs/vibe_coding/alpha-excel/ae2-architecture.md` - 시스템 설계

## 🗺️ 로드맵

### 단기 (진행 중)
- 백테스팅 기능 완성 (weight scaling, 포트폴리오 수익률 계산)
- 다양한 연산자 추가 (TsStd, TsRank, Demean, GroupNeutralize 등)

### 중기
- 더 현실적인 백테스팅 (거래 비용, 슬리피지, 포지션 제약)
- ETL 파이프라인 구축 (데이터 수집 및 전처리 자동화)

### 장기
- alpha-lab 기능 추가 (분석 및 시각화 도구)
- alpha-academia 추가 (학술 논문 구현 및 재현)
- PyPI 배포 및 공개

## 🤝 기여

현재 alpha-excel은 활발히 개발 중인 초기 단계입니다. 안정화 이후 기여를 받을 예정이며, 그 전까지는 외부 기여를 받지 않습니다.

## 📄 라이선스

MIT License
