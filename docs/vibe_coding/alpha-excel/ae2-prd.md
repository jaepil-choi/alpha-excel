# Alpha Excel v2.0 - Product Requirement Document (PRD)

## 1.1. 개요

### 제품 비전

**alpha-excel v2.0**은 v1.0의 핵심 가치를 유지하면서, **성능, 사용성, 메모리 효율성**을 극적으로 개선한 차세대 퀀트 리서치 플랫폼입니다.

**핵심 가치 제안 (v1.0 유지):**

- **Expression-based API**: 선언적이고 조합 가능한 시그널 생성
- **Config-Driven Auto-Loading**: alpha-database 기반 데이터 자동 로딩
- **Auto Universe Masking**: Field 로딩과 결과 반환 시 자동 유니버스 필터링
- **Type-Aware System**: 데이터 타입(numeric, group, weight) 기반 연산

**v2.0 혁신:**

- 🚀 **Eager Execution**: Visitor 패턴 제거, 즉시 평가로 10배 속도 향상
- 💡 **Method-Based API**: `o.ts_mean()` 스타일의 직관적 접근
- 💾 **On-Demand Caching**: 사용자가 원하는 데이터만 캐싱 (메모리 90% 절감)
- 📊 **Stateful Data Model**: 연산 히스토리 자동 추적
- ⚡ **Optimized Group Operations**: NumPy scatter-gather로 5배 빠른 그룹 연산
  (자세한 내용: `docs/research/faster-group-operations.md` 참고)

### v1.0의 문제점

**성능 이슈:**

- Visitor 패턴의 Lazy Execution으로 인한 오버헤드
- 모든 signal, weight, return을 triple-cache에 저장 → 메모리 과다 사용
- 불필요한 데이터 재계산

**사용성 이슈:**

- 연산자를 일일이 import 해야 함

  ```python
  from alpha_excel.ops.timeseries import TsMean, TsStd, TsRank
  from alpha_excel.ops.crosssection import Rank, Demean
  # 매우 불편함
  ```

- DataSource를 외부에서 생성하여 전달해야 함
- Step index로 cache 접근 → 어떤 step이 저장되었는지 파악 어려움
- Scaler를 evaluate() 시점에 적용 → 가중치 계산 타이밍 제어 불가

**설계 이슈:**

- Expression은 stateless, 연산 히스토리 추적 불가
- 모든 중간 결과가 자동 캐싱 → 사용자 제어 불가
- Operator와 Visitor의 책임 분리가 불명확

---

## 1.2. v2.0 핵심 설계 원칙

### 원칙 1: Eager Execution (즉시 평가)

**v1.0 방식 (Lazy):**

```python
# Expression 트리 구축만 하고 평가는 evaluate() 시점
expr = Rank(TsMean(Field('returns'), window=5))
result = ae.evaluate(expr)  # 이 시점에 전체 트리 순회
```

**v2.0 방식 (Eager):**

```python
# 각 연산자 호출 시 즉시 계산
returns = f('returns')  # 즉시 로딩
ma5 = o.ts_mean(returns, window=5)  # 즉시 계산
signal = o.rank(ma5)  # 즉시 계산
```

**장점:**

- 중간 결과를 즉시 확인 가능 (디버깅 용이)
- 불필요한 트리 순회 오버헤드 제거
- 메모리 사용 최적화 (필요한 것만 캐싱)

### 원칙 2: Stateful Data Model (상태 유지 데이터)

**핵심 클래스: `AlphaData`**

```python
class AlphaData:
    _data: pd.DataFrame       # 실제 (T, N) 데이터
    _step_counter: int        # 적용된 연산 수
    _step_history: List[Dict] # 연산 히스토리
    _data_type: str           # 'numeric', 'group', 'weight', etc.
    _cached: bool             # 데이터 캐싱 여부
```

**특징:**

- 연산 히스토리 자동 추적 (`__repr__`으로 표현식 출력)
- 데이터 타입 기반 연산 검증
- 선택적 데이터 캐싱 (`record_output=True`)

### 원칙 3: Stateless Operators (무상태 연산자)

**핵심 클래스: `BaseOperator`**

```python
class BaseOperator:
    output_type: str = 'numeric'
    input_types: List[str] = ['numeric']
    prefer_numpy: bool = False  # 연산자별 최적 데이터 구조 선택

    # prefer_numpy = False (pandas 선호):
    #   - rolling, rank, groupby 등 pandas 내장 최적화 활용
    #   - 예: TsMean, TsStd, Rank
    #
    # prefer_numpy = True (numpy 선호):
    #   - scatter-gather 등 커스텀 알고리즘 구현
    #   - 예: GroupNeutralize, GroupSum

    def compute(self, *data, **params):
        # 순수 계산 로직
        raise NotImplementedError

    def __call__(self, *inputs, record_output=False, **params) -> AlphaData:
        # 1. Input type 검증
        # 2. Data 추출 (to_df or to_numpy)
        # 3. compute() 호출
        # 4. AlphaData로 wrapping
        # 5. Universe masking
        # 6. Step history 업데이트
```

**장점:**

- 연산자는 순수 함수 (테스트 용이)
- Type checking 자동화
- Universe masking 자동 적용

### 원칙 4: Method-Based API (메서드 기반 인터페이스)

**v1.0 방식:**

```python
from alpha_excel.ops.timeseries import TsMean, TsRank
from alpha_excel.ops.crosssection import Rank

expr = Rank(TsMean(Field('returns'), window=5))
```

**v2.0 방식:**

```python
# 모든 연산자가 o.method_name() 형태
o = ae.ops
returns = f('returns')
ma5 = o.ts_mean(returns, window=5)
signal = o.rank(ma5)
```

**장점:**

- Import 불필요
- IDE 자동완성 지원
- 직관적인 메서드 체이닝

### 원칙 5: On-Demand Caching (선택적 캐싱)

**v1.0 방식:**

```python
# 모든 step 자동 캐싱 → 메모리 낭비
result = ae.evaluate(expr)
# signal_cache[0], [1], [2], ... 모두 저장
```

**v2.0 방식:**

```python
# 사용자가 원하는 것만 캐싱
ma5 = o.ts_mean(returns, window=5, record_output=True)  # 캐싱
ma20 = o.ts_mean(returns, window=20)  # 캐싱 안 함
signal = o.subtract(ma5, ma20)
```

**장점:**

- 메모리 사용량 90% 감소
- 중요한 중간 결과만 저장
- 디버깅 시 선택적으로 캐싱 활성화

### 원칙 5-1: Cache Inheritance (캐시 상속)

**요구사항:**

- `record_output=True`로 캐싱된 AlphaData가 다음 연산의 입력이 되면,
- 그 DataFrame은 **출력 AlphaData의 내부 캐시에 저장**됨
- Python 변수 없이도 downstream에서 중간 결과 접근 가능

**사용 예시:**

```python
# Step 1: ma5 캐싱
ma5 = o.ts_mean(returns, window=5, record_output=True)

# Step 2: momentum 계산
# → ma5의 DataFrame이 momentum 내부 캐시에 복사됨
momentum = ma5 - 0.5

# Step 3: signal 계산
# → ma5와 momentum의 DataFrame이 signal 내부 캐시에 복사됨
signal = o.rank(momentum)

# signal에서 이전 step들의 캐싱된 데이터 모두 접근 가능
ma5_data = signal.get_cached_step(1)      # ma5 DataFrame
momentum_data = signal.get_cached_step(2)  # momentum DataFrame

# 캐시되지 않은 step은 None 반환
none_data = signal.get_cached_step(0)  # None (returns는 캐싱 안 함)
```

**장점:**

- 긴 computation chain에서 중간 결과 디버깅 용이
- Python 변수 관리 불필요
- 선택적 캐싱으로 메모리 효율 유지

### 원칙 6: Type-Aware System (타입 인식 시스템)

**데이터 타입 종류:**

- `numeric`: 수치형 데이터 (returns, prices, signals)
- `group`: 범주형 데이터 (industry, sector)
- `weight`: 포트폴리오 가중치
- `port_return`: 포지션별 수익률
- `object`: 기타 (문자열 등)

**타입 기반 동작:**

```python
# Forward fill 전략 (data.yaml에서 설정)
numeric: ffill=0  # 기본값, forward fill 안 함
group: ffill=-1   # 완전 forward fill (월간 → 일간)
weight: ffill=0   # forward fill 안 함

# 연산자 입력 타입 검증
o.group_neutralize(signal, industry)
# → signal: numeric, industry: group 확인
```

**장점:**

- 잘못된 연산 조기 발견
- 데이터 처리 로직 자동화
- 명확한 의미론

---

## 1.3. v1.0 vs v2.0 비교

| 항목 | v1.0 | v2.0 |
|------|------|------|
| **실행 방식** | Lazy (Visitor 패턴) | Eager (즉시 평가) |
| **연산자 접근** | `from ... import TsMean` | `o.ts_mean()` |
| **데이터 모델** | Stateless Expression | Stateful AlphaData (데이터 + 히스토리) |
| **연산자** | Expression에 통합 | Stateless BaseOperator (순수 함수) |
| **캐싱 전략** | 모든 step 자동 캐싱 | 사용자 선택 (`record_output`) |
| **성능** | 느림 (트리 순회) | 10배 빠름 (즉시 실행) |
| **메모리** | 높음 (triple-cache) | 90% 감소 (선택적 캐싱) |
| **타입 시스템** | 없음 | Type-aware (numeric, group, etc.) |
| **DataSource** | 외부 전달 | 내부 생성 |
| **Scaler 적용** | `evaluate()` 시점 | `to_weights()` 시점 |
| **Step 접근** | Index 기반 | AlphaData.history |
| **디버깅** | 어려움 | 쉬움 (중간 결과 즉시 확인) |

---

## 1.4. 사용자 워크플로우 예시

### 워크플로우 1: 기본 시그널 생성

```python
# 1. 초기화
ae = AlphaExcel(start_date='2020-01-01', end_date='2024-12-31')
o = ae.ops
f = ae.field

# 2. Field 로딩
returns = f('returns')

# 3. 시그널 생성 (즉시 실행)
ma5 = o.ts_mean(returns, window=5)
ma20 = o.ts_mean(returns, window=20)
momentum = ma5 - ma20  # Arithmetic operator

# 4. Cross-sectional 순위화
signal = o.rank(momentum)

# 5. 결과 확인
print(signal)  # Expression 출력
df = signal.to_df()  # DataFrame 추출
```

### 워크플로우 2: 백테스트

```python
# 1. 시그널 생성
signal = o.rank(o.ts_mean(returns, window=5))

# 2. Scaler 설정 및 가중치 계산
ae.set_scaler('DollarNeutral')
weights = ae.to_weights(signal)

# 3. 포트폴리오 수익률 계산
port_return = ae.to_portfolio_returns(weights)

# 4. 성과 분석
pnl_df = port_return.to_df()
daily_pnl = pnl_df.sum(axis=1)
cum_pnl = daily_pnl.cumsum()

sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe:.2f}")
```

### 워크플로우 3: 디버깅 및 중간 결과 캐싱

```python
# 중요한 중간 결과만 캐싱
ma5 = o.ts_mean(returns, window=5, record_output=True)  # 캐싱
ma20 = o.ts_mean(returns, window=20)  # 캐싱 안 함
momentum = ma5 - ma20

# 캐싱의 핵심: momentum 내부에 ma5의 DataFrame이 저장됨
# Python 변수 'ma5'를 지워도 momentum에서 복원 가능
del ma5  # 변수 삭제

# momentum의 내부 캐시에서 step 1의 데이터 조회
cached_ma5_data = momentum.get_cached_step(1)  # ma5의 DataFrame
print(cached_ma5_data.shape)  # (T, N)

# Step history로 어떤 step이 캐싱되었는지 확인
for step_info in momentum.history:
    if step_info.get('cached', False):
        print(f"Step {step_info['step']}: {step_info['expr']} is cached")
```

### 워크플로우 4: 그룹 연산 (최적화)

```python
# Group 연산 최적화 (faster-group-operations.md 참고)

# 1. Group field 로딩 (category dtype)
industry = f('fnguide_industry_group')
# → 자동으로 category dtype으로 로딩됨 (메모리 효율 + 속도 향상)

# 2. Sum-based 연산은 NumPy scatter-gather 사용
neutral_signal = o.group_neutralize(returns, industry)  # 5배 빠름

# 3. Rank-based 연산은 Pandas 최적화 활용
ranked = o.group_rank(returns, industry)

# 성능 특성:
# - GroupNeutralize: NumPy scatter-add (100ms vs pandas 500ms)
# - GroupRank: Pandas groupby-rank (최적화된 sorting)
# - 공통: row-by-row iteration 절대 금지
```

**Group field category dtype 요구사항:**

Group 타입 field는 pandas `category` dtype으로 자동 변환:

```python
# Group field는 category dtype으로 로딩
industry = f('fnguide_industry_group')
print(industry.to_df().dtypes)  # dtype: category

# 이유:
# - 메모리 효율: 문자열 대비 50% 감소
# - 그룹 연산 속도: category는 정수 인덱스 기반으로 빠름
# - NumPy scatter-gather 최적화와 호환
```

**참고 문서:**

- 구현 세부사항: `docs/research/faster-group-operations.md`
- NumPy scatter-gather 패턴
- Pandas vs NumPy 성능 비교

### 워크플로우 5: Operator Composition (연산자 조합)

**요구사항:**

- 기존 operator를 조합하여 새 operator 생성
- **반드시 BaseOperator 서브클래스로 구현**
- Registry를 통해 다른 operator 재사용

**구현 예시:**

```python
class TsZscore(BaseOperator):
    output_type = 'numeric'
    input_types = ['numeric']

    def __call__(self, data: AlphaData, window: int, **kwargs) -> AlphaData:
        # Registry를 통해 기존 operator 호출
        mean = self._registry.ts_mean(data, window=window)
        std = self._registry.ts_std(data, window=window)

        # AlphaData의 arithmetic operator 활용
        zscore = (data - mean) / std
        return zscore

# 사용
zscore = o.ts_zscore(returns, window=20)
print(zscore)  # AlphaData(expr='ts_zscore(Field(returns), window=20)')
```

**장점:**

- 코드 재사용 (DRY 원칙)
- 복잡한 시그널을 간단한 API로 제공
- 테스트 용이 (기존 operator의 정확성 활용)

### 워크플로우 6: Long/Short 분리 분석

**Long/Short 분리 수익률:**

```python
# 1. 시그널 생성 및 가중치 계산
signal = o.rank(o.ts_mean(returns, window=5))
ae.set_scaler('DollarNeutral')
weights = ae.to_weights(signal)

# 2. Long positions만 수익률 계산
long_return = ae.to_long_returns(weights)  # AlphaData(type='port_return')

# 3. Short positions만 수익률 계산
short_return = ae.to_short_returns(weights)  # AlphaData(type='port_return')

# 4. 각각 독립 분석
long_pnl = long_return.to_df().sum(axis=1).cumsum()
short_pnl = short_return.to_df().sum(axis=1).cumsum()

# Long/Short 성과 비교
long_sharpe = long_pnl.diff().mean() / long_pnl.diff().std() * np.sqrt(252)
short_sharpe = short_pnl.diff().mean() / short_pnl.diff().std() * np.sqrt(252)

print(f"Long Sharpe: {long_sharpe:.2f}")
print(f"Short Sharpe: {short_sharpe:.2f}")
```

**핵심 메서드:**

- `ae.to_weights(signal)`: Signal → Weights 변환
- `ae.to_portfolio_returns(weights)`: Weights → Position-level returns
- `ae.to_long_returns(weights)`: Positive weights만 사용
- `ae.to_short_returns(weights)`: Negative weights만 사용
- 각각 AlphaData(type='port_return') 반환

---

## 1.5. 다음 단계

1. **아키텍처 설계**: `ae2-architecture.md` 작성 ✅
2. **실험**: `experiments/ae2_*.py` - 핵심 컴포넌트 프로토타입
3. **구현**: `src/alpha_excel/` 패키지 리팩토링
4. **테스트**: 모든 기존 테스트 통과 + 새 기능 테스트
5. **마이그레이션**: Showcase 및 문서 업데이트
6. **배포**: v2.0.0 릴리스
