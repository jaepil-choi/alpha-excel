# 2. 아키텍처 (Architecture)

## 2.0. 개요

alpha-excel은 **Facade**, **Composite**, **Visitor** 디자인 패턴을 기반으로 설계되었습니다. 핵심은 **pandas 기반 데이터 모델**과 **자동 로딩** 메커니즘입니다.

**최근 업데이트 (2025-10-28)**: EvaluateVisitor를 **Single Responsibility Principle (SRP)**에 따라 리팩토링하여 4개의 전문 컴포넌트로 분리했습니다:
- **UniverseMask**: Universe 마스킹 로직
- **StepTracker**: Triple-cache 관리
- **FieldLoader**: 데이터 로딩 및 변환
- **BacktestEngine**: 포트폴리오 수익률 계산

### 2.0.1. 전체 시스템 아키텍처 (SRP 리팩토링 적용)

```mermaid
graph TB
    User[User/Researcher]

    subgraph "AlphaExcel Facade"
        RC[rc: AlphaExcel]
        CTX[(ctx: DataContext)]
    end

    subgraph "Execution Layer (SRP 적용)"
        Visitor[EvaluateVisitor<br/>트리 순회 전담]

        subgraph "Specialized Components"
            UnivMask[UniverseMask<br/>마스킹 로직]
            StepTracker[StepTracker<br/>캐시 관리]
            FieldLoader[FieldLoader<br/>데이터 로딩]
            BacktestEngine[BacktestEngine<br/>백테스트 계산]
        end
    end

    subgraph "Triple Cache (StepTracker 관리)"
        SignalCache[Signal Cache]
        WeightCache[Weight Cache]
        PortReturnCache[Port Return Cache]
    end

    subgraph "Expression Tree"
        Field[Field Nodes]
        Operators[Operator Nodes]
        ExprTree[Expression Tree]
    end

    subgraph "Data Layer"
        DataSource[DataSource]
        Config[config/data.yaml]
        Parquet[(Parquet Files)]
    end

    subgraph "Portfolio Layer"
        Scalers[Weight Scalers]
        Strategies[Scaling Strategies]
    end

    User -->|initialize| RC
    User -->|evaluate| RC
    RC -->|stores| CTX
    RC -->|owns| Visitor
    RC -->|delegates| Scalers

    Visitor -->|uses| UnivMask
    Visitor -->|uses| StepTracker
    Visitor -->|uses| FieldLoader
    Visitor -->|uses| BacktestEngine
    Visitor -->|traverses| ExprTree

    StepTracker -->|manages| SignalCache
    StepTracker -->|manages| WeightCache
    StepTracker -->|manages| PortReturnCache

    FieldLoader -->|loads from| DataSource
    FieldLoader -->|caches in| CTX

    UnivMask -->|applies masking|Visitor
    BacktestEngine -->|uses| UnivMask

    ExprTree -->|contains| Field
    ExprTree -->|contains| Operators

    DataSource -->|reads| Config
    DataSource -->|queries| Parquet

    Scalers -->|uses| Strategies
    Operators -->|compute| Operators
```

### 2.0.2. 데이터 흐름 아키텍처 (SRP 적용)

```mermaid
flowchart LR
    A[Expression Tree] -->|evaluate| B[Visitor<br/>트리 순회]

    B -->|visit Field| C{Is cached?}
    C -->|No| FL[FieldLoader]
    C -->|Yes| E[DataContext]

    FL -->|load_field| DS[DataSource]
    DS -->|DataFrame| FL
    FL -->|apply transform| FL
    FL -->|reindex| FL
    FL -->|INPUT MASK| UM1[UniverseMask]
    UM1 -->|masked data| E
    FL -->|cache| E
    E -->|cached data| B

    B -->|visit Operator| H[Operator.compute]
    H -->|result| UM2[UniverseMask<br/>OUTPUT MASK]
    UM2 -->|masked result| ST[StepTracker]

    ST -->|record_signal| SignalCache
    ST -->|record_weights| WeightCache
    ST -->|record_port_return| PortReturnCache

    ST -->|if scaler provided| BE[BacktestEngine]
    BE -->|compute returns| ST
```

---

## 2.1. 핵심 컴포넌트

### A. `AlphaExcel` (`rc` 인스턴스): Facade 패턴

**역할:** `rc` 인스턴스는 "시그널 캔버스"로서 모든 작업을 위한 **단일 진입점(Facade)**입니다.

**핵심 컴포넌트:**
1. **`rc.ctx` (State):** `DataContext` 인스턴스. `(time, asset)` 형태의 모든 데이터를 pandas DataFrame으로 `dict` 형식으로 저장합니다.
2. **`rc._evaluator` (Executor):** `EvaluateVisitor`의 인스턴스. `Expression` "트리"를 "방문"하여 "평가"합니다.
3. **`rc._data_source` (DataSource):** `alpha_database.DataSource` 인스턴스. Parquet 파일에서 데이터를 로딩합니다.
4. **`rc._universe_mask` (Universe):** `(T, N)` Boolean DataFrame. 모든 연산에 자동 적용됩니다.

**API 메서드:**
- `rc.evaluate(expr)`: Expression 평가 및 자동 데이터 로딩
- `rc.scale_weights(signal, scaler)`: 시그널을 포트폴리오 가중치로 변환
- `rc.get_signal(step)`: 특정 step의 signal 반환
- `rc.get_weights(step)`: 특정 step의 weights 반환
- `rc.get_port_return(step)`: 특정 step의 portfolio return 반환

---

### B. `Expression` 트리: Composite 패턴

**역할:** "수식" 또는 "트리"입니다. **Composite 패턴**으로 데이터 흐름을 표현합니다.

**구조:**
- `Expression`은 모든 노드의 부모 인터페이스입니다.
- **Leaf (잎):** `Field('close')`는 단일 데이터 소스 노드입니다.
- **Composite (가지):** `TsMean(Field('close'), 10)`는 단일 또는 여러 `Expression` 인스턴스를 자식으로 가지는 트리 구조입니다.

**특징:**
- `Expression` 인스턴스는 항상 데이터(`(T, N)` 형태)를 반환할 준비가 되어 있으며, "수식 자체"를 담고 있을 뿐입니다.

**예시:**
```python
from alpha_excel import Field, TsMean, Rank

# Leaf node
returns = Field('returns')

# Composite node
ma5 = TsMean(returns, window=5)

# Nested composite
signal = Rank(ma5)
```

---

### C. `Visitor` 패턴: 순회 및 평가 (SRP 리팩토링 적용)

**역할:** `Expression` 트리를 "방문(visit)"하여 **트리 순회**에만 집중합니다. 다른 책임은 전문 컴포넌트에 위임합니다.

**`EvaluateVisitor`:** `rc` 인스턴스(`rc._evaluator`)가 소유하며, **Single Responsibility Principle**에 따라 다음만 처리합니다:

1. **트리 순회(Traversal):** Depth-first로 모든 노드를 차례로 방문 (핵심 책임)
2. **계산 위임(Delegation):** 각 연산의 `compute()` 메서드를 호출하여 실제 계산 수행

**위임된 책임 (Specialized Components):**
- **UniverseMask**: 입력/출력 universe masking
- **StepTracker**: Signal, weight, port_return 캐시 관리
- **FieldLoader**: 데이터 자동 로딩 및 변환
- **BacktestEngine**: 포트폴리오 수익률 계산

**방문 메서드 (간소화됨):**
- `visit_field()`: FieldLoader에 로딩 위임, UniverseMask로 마스킹, StepTracker에 캐싱
- `visit_operator()`: 계산 수행, UniverseMask로 마스킹, StepTracker에 캐싱
- `visit_constant()`: Constant DataFrame 생성

**리팩토링 후 visit_field (위임 패턴):**
```python
def visit_field(self, field: Field) -> pd.DataFrame:
    # 1. FieldLoader에 로딩 위임
    data = self._field_loader.load_field(field.name)

    # 2. UniverseMask에 INPUT MASKING 위임
    data = self._universe_mask.apply_input_mask(data)

    # 3. StepTracker에 캐싱 위임
    self._step_tracker.record_signal(f"Field_{field.name}", data)
    self._step_tracker.increment_step()

    return data
```

---

### C-1. Specialized Components (SRP 적용)

**리팩토링 배경:** EvaluateVisitor가 6개 이상의 책임을 가지고 있어 SRP 위반 → 전문 컴포넌트로 분리

#### 1. UniverseMask (`core/universe_mask.py`)

**단일 책임:** Universe 마스킹 로직의 중앙화

**주요 메서드:**
- `apply_input_mask(data)`: INPUT MASKING (데이터 진입 시)
- `apply_output_mask(data)`: OUTPUT MASKING (연산 결과 시)

**특징:**
- 멱등성 보장: 중복 마스킹 안전
- Boolean DataFrame 기반
- `data.where(mask, np.nan)` 패턴 통합

```python
class UniverseMask:
    def __init__(self, mask: pd.DataFrame):
        self._mask = mask  # Boolean DataFrame (T, N)

    def apply_input_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 진입 시 마스킹"""
        return data.where(self._mask, np.nan)

    def apply_output_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """연산 결과 마스킹"""
        return data.where(self._mask, np.nan)
```

---

#### 2. StepTracker (`core/step_tracker.py`)

**단일 책임:** Triple-cache 아키텍처 관리

**주요 메서드:**
- `record_signal(name, signal)`: Signal 캐시 저장
- `record_weights(name, weights)`: Weight 캐시 저장
- `record_port_return(name, port_return)`: Portfolio return 캐시 저장
- `reset_signal_cache()`: Signal 캐시 초기화 (새 평가 시)
- `reset_weight_caches()`: Weight/return 캐시만 초기화 (scaler 변경 시)
- `get_signal(step)`, `get_weights(step)`, `get_port_return(step)`: 캐시 조회

**특징:**
- 캐시 계층 분리: Signal (영속적) vs Weight/Return (갱신 가능)
- Step counter 관리
- 타입 안전성: `Dict[int, Tuple[str, pd.DataFrame]]`

```python
class StepTracker:
    def __init__(self):
        self._signal_cache: Dict[int, Tuple[str, pd.DataFrame]] = {}
        self._weight_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}
        self._port_return_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}
        self._step_counter: int = 0

    def record_signal(self, name: str, signal: pd.DataFrame):
        self._signal_cache[self._step_counter] = (name, signal)

    def increment_step(self):
        self._step_counter += 1
```

---

#### 3. FieldLoader (`core/field_loader.py`)

**단일 책임:** 데이터 로딩 및 변환

**주요 메서드:**
- `load_field(field_name, data_type)`: 필드 로딩 (캐싱 포함)
- `set_universe_shape(dates, assets)`: Universe 차원 설정
- `set_date_range(start, end, buffer_start)`: 날짜 범위 설정
- `_apply_forward_fill(data)`: 저빈도 데이터 forward-fill
- `_reindex_to_universe(data)`: Universe 형태로 reindex

**특징:**
- 캐시 확인 → 로딩 → 변환 → reindex → 캐싱 워크플로우
- Forward-fill 변환 지원 (월간 데이터 → 일간)
- DataContext 통합

```python
class FieldLoader:
    def load_field(self, field_name: str, data_type: Optional[str] = None) -> pd.DataFrame:
        # 1. 캐시 확인
        if field_name in self._ctx:
            return self._ctx[field_name]

        # 2. DataSource에서 로딩
        result = self._data_source.load_field(field_name, ...)

        # 3. Forward-fill 적용 (필요시)
        if field_config.get('forward_fill'):
            result = self._apply_forward_fill(result)

        # 4. Buffer 제거
        result = result.loc[self._start_date:]

        # 5. Reindex to universe
        result = self._reindex_to_universe(result)

        # 6. 캐싱
        self._ctx[field_name] = result
        return result
```

---

#### 4. BacktestEngine (`core/backtest_engine.py`)

**단일 책임:** 포트폴리오 수익률 계산

**주요 메서드:**
- `compute_portfolio_returns(weights)`: Position-level returns 계산
- `compute_daily_pnl(port_return)`: Daily PnL 집계
- `compute_metrics(port_return)`: 성과 지표 계산 (Sharpe, drawdown 등)

**특징:**
- Shift-mask 워크플로우 구현
- UniverseMask 통합
- 완전 벡터화

```python
class BacktestEngine:
    def compute_portfolio_returns(self, weights: pd.DataFrame) -> pd.DataFrame:
        # 1. Shift: 다음날 포지션
        weights_shifted = weights.shift(1)

        # 2. Re-mask: Universe 변화 대응
        final_weights = self._universe_mask.apply_output_mask(weights_shifted)

        # 3. Mask returns
        returns_masked = self._universe_mask.apply_output_mask(self._returns_data)

        # 4. Element-wise multiply (T, N shape 유지)
        port_return = final_weights * returns_masked

        return port_return
```

---

### D. Operator 책임 분리

**설계 원칙:** 각 연산(`Expression`)은 순수한 계산 로직만 `compute()` 메서드로 제공합니다.

**Visitor의 역할:**
- 트리 순회 로직
- 데이터 로딩
- 캐싱 관리
- Universe masking

**Operator의 역할:**
- 순수 계산 로직 (`compute()` 메서드)
- Visitor 패턴 준수 (`accept()` 메서드)

**장점:**
1. **단일 책임 원칙(SRP):** Visitor는 순회만, Operator는 계산만
2. **테스트 용이성:** `compute()` 메서드를 독립적으로 테스트 가능
3. **유지보수성:** Visitor가 변경되어도 Operator는 영향 없음
4. **확장성:** 새로운 연산 추가 시 Visitor 수정 불필요

---

## 2.2. 데이터 모델 아키텍처

### A. `DataContext`

`DataContext`는 `alpha-excel`의 핵심 데이터 컨테이너로, 모든 중간 **dict-like 인스턴스**입니다.

**좌표(Coordinates):** 모든 데이터는 `(dates, assets)` 좌표를 공유합니다 (pandas Index 기반).

**데이터 저장(Data Variables):**
- `rc.ctx['name']` 형식으로 DataFrame 접근 가능
- Field 방문 시 자동으로 DataSource에서 로딩하여 저장
- 예시: `rc.ctx['returns']`는 최초 방문 시 자동 로딩됨

**사용 예시:**
```python
# AlphaExcel 초기화
rc = AlphaExcel(data_source=ds, start_date='2024-01-01', end_date='2024-12-31')

# Expression 평가 시 자동으로 ctx에 저장
result = rc.evaluate(Field('returns'))

# 직접 접근 가능
returns_data = rc.ctx['returns']  # pandas DataFrame (T, N)
```

---

### B. 자동 로딩 패턴 (Auto-Loading Pattern)

**Lazy Loading:**
- Field 최초 참조 시에만 DataSource.load_field() 호출
- 이미 로딩된 데이터는 DataContext에서 재사용
- 불필요한 데이터 로딩 방지로 성능 최적화

**자동 로딩 흐름:**
```
Field('returns') 방문
    ↓
DataContext에 'returns' 존재?
    ↓ (No)
DataSource.load_field('returns')
    ↓
Parquet 파일 쿼리
    ↓
Long → Wide 변환 (date × security_id → time × asset)
    ↓
INPUT MASKING (universe 적용)
    ↓
DataContext에 저장 (캐싱)
```

**장점:**
- **효율성:** 필요한 데이터만 로딩
- **투명성:** 사용자는 데이터 로딩을 의식할 필요 없음
- **캐싱:** 동일 Field 재사용 시 즉시 반환

---

### C. Universe Masking

**Investable Universe**는 alpha-excel의 핵심 개념입니다. 모든 데이터와 결과는 universe를 기준으로 필터링됩니다.

#### 1. 핵심 전략: Double Masking

**설계 원칙:** 신뢰 체인(Trust Chain)을 구축하여 모든 데이터가 universe를 보장하도록 합니다.

**INPUT MASKING (입력 마스킹):** `visit_field()`에서 데이터 로딩 시 적용
- 모든 데이터가 시스템에 진입하는 순간 마스킹
- `result = result.where(universe_mask, np.nan)`

**OUTPUT MASKING (출력 마스킹):** `visit_operator()`에서 연산 결과 시 적용
- 연산 결과가 출력되기 전에 universe를 보장하도록 재적용
- Operator 결과에 대한 마스킹 보장

#### 2. 불변성 (Immutability)

- **최초 설정만 가능:** `AlphaExcel(universe=...)` 초기화 시 한 번만 설정
- **변경 불가:** 한번 설정된 universe는 변경 불가 (read-only property로 제공)
- **이유:** 일관성 있는 PnL 분석과 재현성을 위해 고정된 universe 필요

#### 3. None 처리 간소화

**기존 방식 (alpha-canvas):** `if self._universe_mask is not None:` 체크 필요

**alpha-excel 방식:** AlphaExcel 초기화 시 반드시 universe_mask 설정
- 명시적 설정: 사용자가 DataFrame으로 제공
- 자동 파생: `~returns.isna()`로 자동 생성

**결과:** 모든 코드에서 None 처리 간소화됨

#### 4. 아키텍처 흐름

```mermaid
sequenceDiagram
    participant User
    participant AlphaExcel
    participant Visitor
    participant Field
    participant Operator

    User->>AlphaExcel: initialize(universe=mask or None)
    AlphaExcel->>AlphaExcel: derive universe from returns
    AlphaExcel->>Visitor: propagate universe_mask (always set)

    User->>AlphaExcel: evaluate(Expression)
    AlphaExcel->>Visitor: evaluate(Expression)
    Visitor->>Field: visit_field()
    Field->>Visitor: auto-load from DataSource
    Visitor->>Visitor: apply INPUT MASKING (no None check)
    Visitor->>Visitor: cache masked data

    User->>AlphaExcel: evaluate(TsMean(...))
    AlphaExcel->>Visitor: evaluate(TsMean)
    Visitor->>Field: visit_field() [cached]
    Visitor->>Operator: visit_operator()
    Operator->>Operator: compute() [core logic]
    Operator->>Visitor: result
    Visitor->>Visitor: apply OUTPUT MASKING (no None check)
    Visitor->>Visitor: cache masked result
    Visitor->>AlphaExcel: final result
```

#### 5. 멱등성 (Idempotency)

**특징:** Double masking이 멱등성을 가짐 (데이터 중복 마스킹 안전)
```python
data.where(mask, np.nan).where(mask, np.nan) == data.where(mask, np.nan)
```
마스킹된 데이터를 재마스킹해도 결과 동일

#### 6. Auto-Loading 통합

- **자동 로딩된 데이터도 마스킹:** Field 노드에서 데이터 로딩 후 즉시 INPUT MASKING 적용
- **일관성:** 모든 데이터 진입점(최초 로딩, 캐시 로딩)에서 동일하게 마스킹

---

### D. 데이터 로딩 아키텍처 (Data Loading Architecture)

**DataSource 컴포넌트**는 Parquet 파일 기반 데이터 계층을 담당합니다.

#### 1. 핵심 개념

- **파일 기반 DB:** MVP에서는 Parquet 파일을 DB로 사용
- **Config 기반:** `config/data.yaml`에 데이터 소스 정의
- **형식 변환:** alpha_database가 Long 포맷을 Wide 포맷 `(T, N)`으로 변환
- **pandas 출력:** pandas DataFrame으로 출력 (xarray 변환 불필요)

#### 2. 아키텍처 흐름

```
config/data.yaml (Field 정의)
        ↓
DataSource.load_field(field_name)
        ↓
   DuckDB 쿼리
   (Parquet 파일 읽기)
        ↓
   Long → Wide 변환
   (date × security_id → time × asset)
        ↓
   pandas.DataFrame 출력 (T, N)
        ↓
   EvaluateVisitor (INPUT MASKING)
        ↓
   DataContext에 캐시
```

#### 3. Visitor 통합 흐름

**Field 노드 방문 로직:**
1. **캐시 확인:** DataContext에 이미 로딩된 데이터인지 확인
2. **자동 로딩:** 없으면 DataSource.load_field()를 호출하여 Parquet에서 로딩
3. **Reindex:** dates와 assets로 reindex하여 차원 정렬
4. **INPUT MASKING:** Universe mask 적용 (반드시 적용, None 처리 불필요)
5. **캐싱:** DataContext에 저장하여 재사용

**핵심 특징:**
- **Lazy Loading:** 필요할 때만 Parquet 파일 읽기
- **Caching:** 한 번 로딩한 데이터는 DataContext에 캐시
- **Universe Integration:** 데이터 진입 즉시 자동 마스킹

---

## 2.3. 기능별 아키텍처 상세 설명

### F1: 자동 데이터 로딩 (Config-Driven Auto-Loading)

**구현 방식:** DataSource + Auto-loading + Lazy evaluation

**동작 흐름:**
1. `rc` 초기화 시 DataSource 인스턴스를 전달받습니다.
2. `rc.evaluate(TsMean(Field('returns'), 5))` 호출 시:
   - `EvaluateVisitor`가 `Field('returns')` 노드를 방문
   - DataContext에 'returns'가 없으면 자동으로 DataSource.load_field('returns') 호출
   - 로딩된 데이터는 INPUT MASKING 적용
   - DataContext에 캐시
3. 이후 동일 Field 호출 시 캐시에서 재사용

**핵심 구현:** DataSource 통합, Field 자동 로딩, 캐싱

---

### F2: Expression-Only API

**구현 방식:** 명시적 데이터 등록 단계 제거, Expression 평가로 완결

**비교:**
```python
# 기존 방식 (명시적 등록 필요)
rc.add_data('returns', Field('returns'))
rc.add_data('ma5', TsMean(Field('returns'), 5))
result = rc.data['ma5']

# alpha-excel 방식 (Expression only)
result = rc.evaluate(TsMean(Field('returns'), 5))
```

**핵심 구현:** evaluate() 메서드만으로 완결

---

### F3: Triple-Cache Architecture

**구현 방식:** 캐시 기반 step 추적, 계층적 재계산 아키텍처

#### 1. 핵심 개념

캐시 구조가 견고하고(robust), 예측 가능하며(predictable), 타입 안전합니다(type-safe).

#### 2. Stateful Visitor

`rc._evaluator` (Visitor)는 **"Stateful(상태 유지)"** 인스턴스입니다.

#### 3. Triple-Cache 구조

`EvaluateVisitor`는 PnL 추적을 위해 세 가지 캐시를 관리합니다:

**Signal Cache (영속적):**
```python
_signal_cache: dict[int, tuple[str, pd.DataFrame]]
```
- 키: **캐시 step 좌표** (0부터 순차)
- 값: `(노드_이름, signal_DataFrame)` 튜플
- **영속성:** scaler 변경 시에도 재사용 (불변성)

**Weight Cache (갱신 가능):**
```python
_weight_cache: dict[int, tuple[str, Optional[pd.DataFrame]]]
```
- 키: **캐시 step 좌표** (0부터 순차)
- 값: `(노드_이름, weights_DataFrame or None)` 튜플
- **갱신성:** scaler 변경 시 재계산됨
- **Optional:** scaler가 제공되지 않으면 None으로 저장

**Portfolio Return Cache (갱신 가능):**
```python
_port_return_cache: dict[int, tuple[str, Optional[pd.DataFrame]]]
```
- 키: **캐시 step 좌표** (0부터 순차)
- 값: `(노드_이름, port_return_DataFrame or None)` 튜플
- **갱신성:** scaler 변경 시 재계산됨
- **Shape:** `(T, N)` 형태 (position-level returns)

#### 4. 캐싱 로직

Visitor는 `Expression` 트리를 **Depth-first 순회**하면서 **각 노드를 평가할 때마다 중간 결과를 순차적으로 캐시**합니다.

**예시 Expression:** `Rank(TsMean(Field('returns'), 3))`
- `signal_cache[0]` = `('Field_returns', DataFrame(...))`
- `signal_cache[1]` = `('TsMean', DataFrame(...))`
- `signal_cache[2]` = `('Rank', DataFrame(...))`
- scaler가 제공되면 각 step의 `weight_cache[i]`와 `port_return_cache[i]`도 저장

#### 5. Triple-Cache 사용 패턴

```python
# 최초 평가 with scaler
result = rc.evaluate(expr, scaler=DollarNeutralScaler())

# 각 step의 signal, weight, port_return 접근
for step in range(len(rc._evaluator._signal_cache)):
    name, signal = rc._evaluator.get_cached_signal(step)
    name, weights = rc._evaluator.get_cached_weights(step)
    name, port_return = rc._evaluator.get_cached_port_return(step)

    if weights is not None:
        daily_pnl = port_return.sum(axis=1)
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        print(f"Step {step} ({name}): Sharpe = {sharpe:.2f}")

# Scaler 변경 (재평가하지 않고 signal 재사용)
result = rc.evaluate(expr, scaler=GrossNetScaler(2.0, 0.3))
# signal_cache 동일 유지, weight_cache와 port_return_cache만 재계산
```

#### 6. Triple-Cache 핵심 장점

- **효율성:** Scaler 변경 시 signal 재계산 불필요 (캐시 재사용)
- **추적성:** 모든 step에서 signal, weight, port_return 개별 접근 가능
- **연구 친화성:** 각 signal의 독립적인 성능을 비교 검증 가능
- **분석 준비:** 단계별 attribution 분석 가능
- **선택성:** weight와 port_return 캐싱은 선택적 (scaler 없으면 None)

---

### F4: 가중치 스케일링 (Weight Scaling)

**구현 방식:** Strategy Pattern 기반의 가중치 정규화 전략

#### 1. 아키텍처 설계

**Strategy Pattern** 기반의 가중치 정규화 전략을 설계합니다.

```mermaid
classDiagram
    class WeightScaler {
        <<abstract>>
        +scale(signal: DataFrame) DataFrame
        #_validate_signal(signal)
    }

    class GrossNetScaler {
        -target_gross: float
        -target_net: float
        -L_target: float
        -S_target: float
        +scale(signal: DataFrame) DataFrame
        #_scale_single_period(signal_slice)
    }

    class DollarNeutralScaler {
        +scale(signal: DataFrame) DataFrame
    }

    class LongOnlyScaler {
        -target_long: float
        +scale(signal: DataFrame) DataFrame
        #_scale_single_period(signal_slice)
    }

    class AlphaExcel {
        +scale_weights(signal, scaler) DataFrame
    }

    WeightScaler <|-- GrossNetScaler
    WeightScaler <|-- LongOnlyScaler
    GrossNetScaler <|-- DollarNeutralScaler
    AlphaExcel ..> WeightScaler : uses
```

#### 2. 대표 전략: GrossNetScaler

**제약 조건:**
- 입력: `target_gross` ($G$), `target_net` ($N$)
- 출력: Long/Short 목표 계산
  $$L_{\text{target}} = \frac{G + N}{2}, \quad S_{\text{target}} = \frac{G - N}{2}$$

**스케일링 흐름 (완전 벡터화):**
```
Signal (T, N) with arbitrary values
    ↓
Separate positive/negative (vectorized)
    ↓
Normalize: s_pos/sum(s_pos), |s_neg|/sum(|s_neg|)  [handle 0/0 → NaN → 0]
    ↓
Apply targets: weights = norm_pos * L_target - norm_neg * |S_target|
    ↓
Calculate actual_gross per row (vectorized)
    ↓
Scale to meet target: weights * (target_gross / actual_gross)
    ↓
Convert computational NaN to 0 (fillna before universe mask)
    ↓
Apply universe mask (preserves signal NaN)
    ↓
Weights (T, N) satisfying gross constraint
```

#### 3. Facade 통합

```python
class AlphaExcel:
    def scale_weights(
        self,
        signal: pd.DataFrame,
        scaler: WeightScaler
    ) -> pd.DataFrame:
        """Scale signal to portfolio weights.

        Args:
            signal: DataFrame with signal values
            scaler: WeightScaler strategy (REQUIRED - no default)

        Returns:
            (T, N) DataFrame with portfolio weights
        """
        # Apply scaling strategy (delegation)
        weights = scaler.scale(signal)
        return weights
```

---

### F5: 백테스트 (Backtesting)

**구현 방식:** Shift-mask 워크플로우, Position-level returns, Triple-cache 통합

#### 1. Shift-Mask 워크플로우

```python
# 백테스트 로직 (완전 벡터화):
# 1. Shift: 다음날 signal로 포지션 형성
weights_shifted = weights.shift(1)  # pandas shift (axis=0)

# 2. Re-mask: 최신 universe로 재마스킹 (universe 변화 가능성 대응)
final_weights = weights_shifted.where(universe_mask)

# 3. Element-wise multiply: 포지션별 수익 계산
port_return = final_weights * returns  # (T, N) shape 유지!
```

#### 2. Position-Level Returns (Shape 유지)

- Portfolio return은 `(T, N)` shape으로 유지 (자산 레벨 보존)
- Winner/loser 분석 가능 (어떤 자산이 PnL 기여?)
- On-demand 집계: `daily_pnl = port_return.sum(axis=1)`

#### 3. API 메서드

- `rc.get_port_return(step)`: Position-level returns `(T, N)` DataFrame
- `rc.get_daily_pnl(step)`: Daily PnL `(T,)` Series (on-demand 집계)
- `rc.get_cumulative_pnl(step)`: Cumulative PnL `(T,)` Series (cumsum 적용)

#### 4. Re-Masking의 중요성 (NaN Pollution 방지)

**문제:** 과거 시점에 universe에서 제외된 자산에 weight가 NaN → `NaN * return = NaN` → PnL이 NaN
**해결책:** Shift 후 최신 universe로 re-mask → 제거된 자산은 포지션 0으로
**효과:** 모든 시뮬레이션에서 NaN pollution 방지 확인

---

### F6: Serialization

**구현 방식:** SerializationVisitor, DeserializationVisitor, DependencyExtractor

**주요 컴포넌트:**

1. **SerializationVisitor:** Expression → JSON-compatible dict
   - 트리 구조를 재귀 순회
   - Nested structure 유지

2. **DeserializationVisitor:** dict → Expression 복원
   - Type dispatch로 올바른 Expression 생성
   - Recursive reconstruction

3. **DependencyExtractor:** Field 의존성 추출
   - 데이터 로딩 준비 용이
   - Lineage 추적

**사용 패턴:**
```python
# Serialization
expr = Rank(TsMean(Field('returns'), 5))
expr_dict = expr.to_dict()

# Deserialization
expr_loaded = Expression.from_dict(expr_dict)

# Dependency extraction
deps = expr.get_field_dependencies()  # ['returns']
```

---

## 2.4. 설계 원칙 및 근거

### 2.4.1. 왜 pandas DataFrame인가?

1. **친숙성(Familiarity):** 리서처들 대부분이 pandas에 익숙함
2. **생태계(Ecosystem):** 풍부한 pandas 라이브러리 활용 가능
3. **간결성(Simplicity):** xarray 학습 곡선 제거
4. **성능(Performance):** pandas의 벡터화 연산으로 충분히 빠름

### 2.4.2. 왜 자동 로딩인가?

1. **간결성(Conciseness):** add_data() 제거로 코드량 50% 감소
2. **명확성(Clarity):** Expression에만 집중, 데이터 흐름 자동
3. **효율성(Efficiency):** Lazy loading + caching으로 필요한 데이터만 로딩
4. **직관성(Intuitiveness):** "필드를 참조하면 자동으로 로딩" 자연스러운 경험

### 2.4.3. 왜 Universe Mask None 처리를 간소화했는가?

1. **보장(Guarantee):** AlphaExcel 초기화 시 반드시 universe_mask 설정
2. **간결성(Simplicity):** 모든 코드에서 중복 체크 제거
3. **성능(Performance):** 조건 분기 제거로 약간의 성능 향상
4. **명확성(Clarity):** 코드 읽기 쉬워짐

### 2.4.4. 왜 Triple-Cache인가?

1. **효율성(Efficiency):** Scaler 변경 시 signal 재계산 불필요
2. **추적성(Traceability):** 모든 step에서 signal, weight, port_return 접근
3. **연구 친화성(Research-Friendly):** 각 시그널의 성능을 비교 검증 가능
4. **분석 준비(Analytics-Ready):** 단계별 attribution 분석 가능

---
