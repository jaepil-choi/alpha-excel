# 3. Implementation Guide

이 문서는 alpha-excel의 실제 구현 세부사항, 코드 패턴, 사용 예시를 다룹니다.

## 3.0. 구현 상태 요약 (Implementation Status Summary)

**최종 업데이트**: 2025-10-28

### ✅ **핵심 기능 구현 완료**

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| **F1: Auto Data Loading** | ✅ DONE | `core/field_loader.py` | Field auto-loading via FieldLoader |
| **F2: Expression-Only API** | ✅ DONE | `core/facade.py` | Direct evaluate() without add_data() |
| **F3: Triple-Cache** | ✅ DONE | `core/step_tracker.py` | Signal, weight, port_return caching |
| **F4: Portfolio (Weights)** | ✅ DONE | `portfolio/` | Strategy pattern weight scalers |
| **F5: Backtesting** | ✅ DONE | `core/backtest_engine.py` | Shift-mask workflow, position returns |
| **F6: Serialization** | ✅ DONE | `core/serialization.py` | Expression to/from dict, dependencies |
| **Universe Masking** | ✅ DONE | `core/universe_mask.py` | Centralized masking (double masking) |
| **SRP Refactoring** | ✅ DONE | `core/visitor.py` + 4 components | Visitor focuses on tree traversal only |
| **Arithmetic Operators** | ✅ DONE | `ops/arithmetic.py` | Add, Subtract, Multiply, Divide, Negate, Abs |
| **Logical Operators** | ✅ DONE | `ops/logical.py` | Comparisons, And, Or, Not, IsNan |
| **Time-Series Operators** | ✅ CORE DONE | `ops/timeseries.py` | Rolling, shift, stats (15 ops implemented) |

### ✅ **검증 완료**

* ✅ **통합 테스트 통과** (`test_alpha_excel.py`)
* ✅ **Showcase 동작** (`showcase_alpha_excel.py`, `showcase_alpha_excel_comprehensive.py`)
* ✅ **Weight caching showcase** (`showcase_weight_caching.py`)
* ✅ **완전 벡터화** (vectorized pandas operations)
* ✅ **Triple-cache 검증** (signal, weight, port_return)

### 🔜 **향후 구현 예정**

| 기능 | 우선순위 | 설명 |
|------|---------|------|
| **Label Quantile** | HIGH | Fama-French Factor 스타일 그룹핑 |
| **String Universe** | MEDIUM | 'univ100', 'univ200' 등 사전 정의 유니버스 |
| **Group Operations** | MEDIUM | GroupNeutralize, GroupDemean (industry 등) |

---

## 3.1. 프로젝트 구조

```text
alpha-canvas/
├── config/                      # 설정 파일 (alpha-database 연동)
│   └── data.yaml               # 데이터 소스 정의
├── src/
│   └── alpha_excel/
│       ├── __init__.py         # Public API exports
│       ├── core/
│       │   ├── facade.py       # AlphaExcel (rc) 클래스 구현
│       │   ├── expression.py   # Expression 기본 인터페이스
│       │   ├── visitor.py      # EvaluateVisitor (SRP 적용, 트리 순회 전담)
│       │   ├── data_model.py   # DataContext (dict-like pandas storage)
│       │   ├── serialization.py # Serialization/Deserialization visitors
│       │   ├── universe_mask.py # [NEW] UniverseMask (마스킹 로직)
│       │   ├── step_tracker.py  # [NEW] StepTracker (triple-cache 관리)
│       │   ├── field_loader.py  # [NEW] FieldLoader (데이터 로딩/변환)
│       │   └── backtest_engine.py # [NEW] BacktestEngine (백테스트 계산)
│       ├── ops/                # 연산자
│       │   ├── __init__.py
│       │   ├── timeseries.py   # TsMean, TsStd, etc.
│       │   ├── crosssection.py # Rank, Demean, etc.
│       │   ├── group.py        # GroupNeutralize, GroupRank, etc.
│       │   ├── arithmetic.py   # Add, Subtract, Multiply, Divide, etc.
│       │   ├── logical.py      # And, Or, Not, Equals, etc.
│       │   └── constants.py    # Constant value operator
│       └── portfolio/          # 포트폴리오 전략
│           ├── __init__.py
│           ├── base.py         # WeightScaler 추상 베이스 클래스
│           └── strategies.py   # GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
├── showcase/                   # Showcase 스크립트
├── tests/                      # 테스트
└── docs/
    └── vibe_coding/
        └── alpha-excel/
            ├── ae-prd.md
            ├── ae-architecture.md
            └── ae-implementation.md   # 이 문서
```

**핵심 설계 원칙 (SRP 적용):**
* `core/expression.py`, `ops/*`: Expression 인터페이스와 연산자 정의
* `core/facade.py`: AlphaExcel facade (단일 진입점)
* `core/visitor.py`: Expression 트리 순회 전담 (SRP)
* **`core/universe_mask.py`**: Universe 마스킹 로직 중앙화
* **`core/step_tracker.py`**: Triple-cache 관리 전담
* **`core/field_loader.py`**: 데이터 로딩 및 변환 전담
* **`core/backtest_engine.py`**: 백테스트 계산 전담
* `core/serialization.py`: Expression 직렬화/역직렬화
* `portfolio/*`: Strategy Pattern 기반 가중치 스케일링

---

## 3.2. 핵심 컴포넌트 구현 상세

### 3.2.1. AlphaExcel 초기화

```python
from alpha_database import DataSource
from alpha_excel import AlphaExcel

# DataSource 초기화 (alpha_database 사용)
ds = DataSource('config')

# AlphaExcel 초기화 (returns 자동 로딩)
rc = AlphaExcel(
    data_source=ds,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 커스텀 universe 지정
universe_mask = (price > 5.0) & (volume > 100000)
rc = AlphaExcel(
    data_source=ds,
    start_date='2024-01-01',
    end_date='2024-12-31',
    universe=universe_mask
)
```

**구현 세부사항:**

* `AlphaExcel.__init__()`는 `DataSource` 인스턴스를 필수로 받습니다
* `start_date`, `end_date`는 날짜 범위를 지정 (데이터 쿼리 범위)
* `universe` 파라미터는 선택적 (None이면 returns에서 자동 파생)
* Returns 데이터는 초기화 시 자동 로딩 (`_load_returns()`)

---

### 3.2.2. 핵심 데이터 모델 구현 (Core Data Model Implementation)

#### A. `DataContext` 구현 (`data_model.py`)

```python
import pandas as pd
from typing import Dict

class DataContext:
    """Dict-like container for pandas DataFrames with shared (dates, assets) index.

    All DataFrames stored in DataContext must have:
    - Index: pd.DatetimeIndex (dates)
    - Columns: pd.Index (assets)

    This ensures all data shares the same coordinate system.
    """

    def __init__(self, dates: pd.DatetimeIndex, assets: pd.Index):
        """Initialize DataContext with shared coordinate system.

        Args:
            dates: Time index (pd.DatetimeIndex)
            assets: Asset index (pd.Index)
        """
        self._dates = dates
        self._assets = assets
        self._data: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, name: str) -> pd.DataFrame:
        """Get DataFrame by name."""
        if name not in self._data:
            raise KeyError(f"Data '{name}' not found in DataContext")
        return self._data[name]

    def __setitem__(self, name: str, value: pd.DataFrame) -> None:
        """Set DataFrame by name.

        Args:
            name: Data variable name
            value: pandas DataFrame with (dates, assets) index/columns
        """
        # Validate shape
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Value must be pandas DataFrame, got {type(value)}")

        # Store (coordinates checked by caller)
        self._data[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if name exists in DataContext."""
        return name in self._data

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Get time index."""
        return self._dates

    @property
    def assets(self) -> pd.Index:
        """Get asset index."""
        return self._assets
```

**핵심 특징:**
* ❌ **No xarray**: 순수 dict 기반 DataFrame 저장소
* ✅ **Shared coordinates**: 모든 DataFrame이 동일한 (dates, assets) 좌표계
* ✅ **Dict-like interface**: `ctx['name']` 형식으로 접근
* ✅ **Type safety**: DataFrame만 저장 가능

---

#### B. `AlphaExcel` 클래스 구현 (`facade.py`)

**핵심 파트 1: 초기화 및 Returns 로딩**

```python
class AlphaExcel:
    def __init__(
        self,
        data_source: 'DataSource',
        start_date: str,
        end_date: Optional[str] = None,
        universe: Optional[Union[str, pd.DataFrame]] = None
    ):
        """Initialize AlphaExcel with DataSource and date range.

        Args:
            data_source: DataSource instance (MANDATORY)
            start_date: Start date (MANDATORY)
            end_date: End date (optional, None = all data from start_date)
            universe: Optional universe (str/DataFrame/None)
        """
        # Store parameters
        self._data_source = data_source
        self.start_date = start_date
        self.end_date = end_date

        # Load returns FIRST (mandatory)
        returns_data = self._load_returns()

        # Handle universe parameter
        if universe is not None:
            if isinstance(universe, str):
                raise NotImplementedError(
                    "String universe (e.g., 'univ100') not yet implemented"
                )
            elif isinstance(universe, pd.DataFrame):
                universe_mask = universe
                dates = pd.DatetimeIndex(universe_mask.index)
                assets = pd.Index(universe_mask.columns)
                returns_data = returns_data.reindex(index=dates, columns=assets)
            else:
                raise TypeError(f"universe must be str or DataFrame, got {type(universe)}")
        else:
            # Derive universe from returns
            dates = pd.DatetimeIndex(returns_data.index)
            assets = pd.Index(returns_data.columns)
            universe_mask = ~returns_data.isna()

        # Create data context
        self.ctx = DataContext(dates, assets)
        self.ctx['returns'] = returns_data

        # Initialize evaluator
        self._evaluator = EvaluateVisitor(self.ctx, data_source=data_source)

        # Store universe mask
        self._universe_mask = universe_mask

        # Initialize specialized components in evaluator (SRP 적용)
        self._evaluator.initialize_components(
            universe_mask_df=universe_mask,
            returns_data=returns_data,
            start_date=start_date,
            end_date=end_date,
            buffer_start_date=self._buffer_start_date
        )

    def _load_returns(self) -> pd.DataFrame:
        """Load returns data from DataSource.

        Returns:
            Returns DataFrame with (time, asset) dimensions

        Raises:
            ValueError: If 'returns' field not found
        """
        try:
            loaded_data = self._data_source.load_field(
                'returns',
                start_date=self.start_date,
                end_date=self.end_date if self.end_date else self.start_date
            )
            # Convert to pandas if necessary
            if hasattr(loaded_data, 'to_pandas'):
                returns_data = loaded_data.to_pandas()
            else:
                returns_data = loaded_data
        except KeyError:
            raise ValueError(
                "Return data is mandatory. Missing 'returns' field in config/data.yaml"
            )

        return returns_data
```

**핵심 파트 2: Expression 평가 (No add_data())**

```python
def evaluate(self, expr: Expression, scaler: Optional['WeightScaler'] = None) -> pd.DataFrame:
    """Evaluate Expression and return result.

    Args:
        expr: Expression to evaluate
        scaler: Optional WeightScaler for portfolio construction

    Returns:
        pandas DataFrame result

    Example:
        >>> result = rc.evaluate(TsMean(Field('returns'), 5))
        >>>
        >>> # With weight scaling and backtesting
        >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
    """
    return self._evaluator.evaluate(expr, scaler)
```

**핵심 파트 3: 캐시 결과 접근**

```python
def get_signal(self, step: int) -> tuple[str, pd.DataFrame]:
    """Get cached signal for a specific step."""
    return self._evaluator.get_cached_signal(step)

def get_weights(self, step: int) -> tuple[str, Optional[pd.DataFrame]]:
    """Get cached portfolio weights for a specific step."""
    return self._evaluator.get_cached_weights(step)

def get_port_return(self, step: int) -> tuple[str, Optional[pd.DataFrame]]:
    """Get cached position-level portfolio returns (T, N)."""
    return self._evaluator.get_cached_port_return(step)

def get_daily_pnl(self, step: int) -> Optional[pd.Series]:
    """Get daily PnL (T,) aggregated across assets."""
    _, port_return = self.get_port_return(step)
    if port_return is None:
        return None
    daily_pnl = port_return.sum(axis=1)
    return daily_pnl

def get_cumulative_pnl(self, step: int) -> Optional[pd.Series]:
    """Get cumulative PnL (T,) using cumsum."""
    daily_pnl = self.get_daily_pnl(step)
    if daily_pnl is None:
        return None
    cumulative_pnl = daily_pnl.cumsum()
    return cumulative_pnl
```

**핵심 특징:**
* ❌ **No add_data()**: Expression 평가만으로 완결
* ✅ **Auto-loading**: Field 참조 시 visitor가 자동 로딩
* ✅ **Universe always set**: None 처리 불필요 (초기화 시 보장)
* ✅ **Triple-cache access**: get_signal, get_weights, get_port_return, get_daily_pnl, get_cumulative_pnl

---

#### C. `EvaluateVisitor` 구현 (SRP 적용) (`visitor.py`)

**컴포넌트 초기화 (initialize_components)**

```python
def initialize_components(
    self,
    universe_mask_df: pd.DataFrame,
    returns_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    buffer_start_date: str
):
    """Initialize specialized components after construction.

    This is called by AlphaExcel facade after visitor creation.
    """
    # Initialize UniverseMask
    self._universe_mask = UniverseMask(universe_mask_df)

    # Initialize StepTracker
    self._step_tracker = StepTracker()

    # Initialize FieldLoader if not already done
    if self._field_loader is not None:
        self._field_loader.set_universe_shape(
            universe_mask_df.index,
            universe_mask_df.columns
        )
        self._field_loader.set_date_range(start_date, end_date, buffer_start_date)

    # Initialize BacktestEngine
    self._backtest_engine = BacktestEngine(returns_data, self._universe_mask)
```

**Field 방문 (visit_field) - 위임 패턴**

```python
def visit_field(self, node: Field) -> pd.DataFrame:
    """Visit Field node: delegate to FieldLoader.

    Workflow (SRP 적용):
    1. FieldLoader에 로딩 위임 (캐싱, 변환, reindex 모두 포함)
    2. UniverseMask에 INPUT MASKING 위임
    3. StepTracker에 signal 캐싱 위임
    4. Return masked result
    """
    # Step 1: FieldLoader에 로딩 위임
    if self._field_loader is None:
        if node.name not in self._ctx:
            raise RuntimeError(f"Field '{node.name}' not found in context")
        result = self._ctx[node.name]
    else:
        result = self._field_loader.load_field(node.name, node.data_type)

    # Step 2: UniverseMask에 INPUT MASKING 위임
    result = self._universe_mask.apply_input_mask(result)

    # Step 3: StepTracker에 캐싱 위임
    self._step_tracker.record_signal(f"Field_{node.name}", result)

    # If scaler provided, compute weights and port_return
    if self._scaler is not None:
        self._cache_weights_and_returns(f"Field_{node.name}", result)

    # Step counter increment
    self._step_tracker.increment_step()

    return result
```

**연산자 방문 (visit_operator) - 위임 패턴**

```python
def visit_operator(self, node: Expression) -> pd.DataFrame:
    """Visit operator node with OUTPUT MASKING.

    Workflow (SRP 적용):
    1. Traverse children (depth-first)
    2. Delegate to operator's compute() method
    3. UniverseMask에 OUTPUT MASKING 위임
    4. StepTracker에 캐싱 위임
    5. Return masked result
    """
    # Step 1 & 2: Traverse and compute
    # (traverse logic unchanged - detect child/left/right)
    result = node.compute(child_result, visitor=self)

    # Step 3: UniverseMask에 OUTPUT MASKING 위임
    result = self._universe_mask.apply_output_mask(result)

    # Step 4: StepTracker에 캐싱 위임
    operator_name = type(node).__name__
    self._cache_signal_weights_and_returns(operator_name, result)

    return result
```

**캐싱 로직 (cache_signal_weights_and_returns) - 위임 패턴**

```python
def _cache_signal_weights_and_returns(self, name: str, signal: pd.DataFrame):
    """Cache signal, weights, and returns using StepTracker and BacktestEngine.

    SRP 적용: 위임만 수행, 실제 계산은 전문 컴포넌트에서
    """
    # StepTracker에 signal 기록
    self._step_tracker.record_signal(name, signal)

    # If scaler provided
    if self._scaler is not None:
        try:
            # Compute weights using scaler
            weights = self._scaler.scale(signal)
            self._step_tracker.record_weights(name, weights)

            # BacktestEngine에 portfolio return 계산 위임
            if self._backtest_engine is not None:
                port_return = self._backtest_engine.compute_portfolio_returns(weights)
                self._step_tracker.record_port_return(name, port_return)
            else:
                self._step_tracker.record_port_return(name, None)

        except Exception as e:
            # If scaling fails, cache None
            self._step_tracker.record_weights(name, None)
            self._step_tracker.record_port_return(name, None)

    # Increment step counter
    self._step_tracker.increment_step()
```

**핵심 특징 (SRP 적용 후):**
* ✅ **Single Responsibility**: Visitor는 트리 순회만, 나머지는 전문 컴포넌트에 위임
* ✅ **Testability**: 각 컴포넌트를 독립적으로 테스트 가능
* ✅ **Maintainability**: 책임 분리로 코드 이해 및 수정 용이
* ✅ **Reusability**: 컴포넌트를 다른 컨텍스트에서 재사용 가능

---

### 3.2.3. Serialization 구현 (`serialization.py`)

**SerializationVisitor: Expression → dict**

```python
class SerializationVisitor:
    """Convert Expression tree to JSON-compatible dictionary."""

    def visit_field(self, node: Field) -> Dict[str, Any]:
        """Serialize Field node."""
        return {
            'type': 'Field',
            'name': node.name
        }

    def visit_operator(self, node: Expression) -> Dict[str, Any]:
        """Serialize operator node with type dispatch."""
        node_type = type(node).__name__
        result = {'type': node_type}

        # Handle different operator signatures
        if hasattr(node, 'left') and hasattr(node, 'right'):
            # Binary operator
            result['left'] = node.left.accept(self)
            result['right'] = node.right.accept(self)
        elif hasattr(node, 'child'):
            # Unary operator
            result['child'] = node.child.accept(self)

        # Add parameters (window, bins, labels, group_by, mask)
        for attr in ['window', 'bins', 'labels', 'group_by', 'mask']:
            if hasattr(node, attr):
                result[attr] = getattr(node, attr)

        return result

    def visit_constant(self, node: Constant) -> Dict[str, Any]:
        """Serialize Constant node."""
        return {
            'type': 'Constant',
            'value': node.value
        }
```

**DeserializationVisitor: dict → Expression**

```python
class DeserializationVisitor:
    """Reconstruct Expression from dictionary."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Expression:
        """Deserialize dictionary to Expression.

        Args:
            data: Serialized Expression dictionary

        Returns:
            Reconstructed Expression object
        """
        node_type = data['type']

        # Field node
        if node_type == 'Field':
            return Field(data['name'])

        # Constant node
        if node_type == 'Constant':
            return Constant(data['value'])

        # Operator nodes (type dispatch)
        operator_class = _get_operator_class(node_type)

        # Reconstruct children
        if 'left' in data and 'right' in data:
            # Binary operator
            left = DeserializationVisitor.from_dict(data['left'])
            right = DeserializationVisitor.from_dict(data['right'])
            return operator_class(left, right)
        elif 'child' in data:
            # Unary operator
            child = DeserializationVisitor.from_dict(data['child'])

            # Extract parameters
            kwargs = {}
            for key in ['window', 'bins', 'labels', 'group_by', 'mask']:
                if key in data:
                    kwargs[key] = data[key]

            return operator_class(child, **kwargs)
        else:
            raise ValueError(f"Unknown operator structure for {node_type}")
```

**DependencyExtractor: Expression → List[str]**

```python
class DependencyExtractor:
    """Extract field dependencies from Expression tree."""

    @staticmethod
    def extract(expr: Expression) -> List[str]:
        """Extract all Field dependencies.

        Args:
            expr: Expression to analyze

        Returns:
            List of unique field names
        """
        visitor = _DependencyVisitor()
        expr.accept(visitor)
        return sorted(set(visitor.fields))


class _DependencyVisitor:
    """Internal visitor for dependency extraction."""

    def __init__(self):
        self.fields = []

    def visit_field(self, node: Field) -> None:
        """Record field name."""
        self.fields.append(node.name)

    def visit_operator(self, node: Expression) -> None:
        """Recurse into children."""
        if hasattr(node, 'left'):
            node.left.accept(self)
        if hasattr(node, 'right'):
            node.right.accept(self)
        if hasattr(node, 'child'):
            node.child.accept(self)

    def visit_constant(self, node: Constant) -> None:
        """No dependencies for constants."""
        pass
```

**Expression 클래스 메서드**

```python
class Expression:
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Expression to dictionary."""
        visitor = SerializationVisitor()
        return self.accept(visitor)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Expression':
        """Deserialize dictionary to Expression."""
        return DeserializationVisitor.from_dict(data)

    def get_field_dependencies(self) -> List[str]:
        """Extract field dependencies."""
        return DependencyExtractor.extract(self)
```

**핵심 특징:**
* ✅ **JSON-compatible**: 모든 Expression이 JSON으로 저장 가능
* ✅ **Round-trip**: to_dict() → from_dict() 왕복 변환
* ✅ **Dependency tracking**: 필요한 데이터 필드 추출
* ✅ **Type safety**: 타입 기반 디스패치로 안전한 복원

---

## 3.3. 연산자 구현 패턴

### 3.3.1. Arithmetic Operators (`ops/arithmetic.py`)

```python
@dataclass(eq=False)
class Add(Expression):
    """Addition operator (A + B)."""
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Element-wise addition - pandas native."""
        return left_result + right_result
```

**핵심 패턴:**
* ✅ **Dataclass**: `@dataclass(eq=False)` for Expression nodes
* ✅ **Generic accept**: All operators use `visitor.visit_operator(self)`
* ✅ **Operator-owned compute**: Calculation logic in `compute()` method
* ✅ **pandas operations**: Direct pandas DataFrame operations

---

### 3.3.2. Time-Series Operators (`ops/timeseries.py`)

```python
@dataclass(eq=False)
class TsMean(Expression):
    """Time-series rolling mean."""
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling mean using pandas."""
        return child_result.rolling(window=self.window, min_periods=self.window).mean()
```

**핵심 패턴:**
* ✅ **Window parameter**: Stored as dataclass field
* ✅ **Pandas rolling**: Direct use of pandas `.rolling()` API
* ✅ **min_periods**: Set to `window` for consistency (NaN until full window)

---

### 3.3.3. Cross-Sectional Operators (`ops/crosssection.py`)

```python
@dataclass(eq=False)
class Rank(Expression):
    """Cross-sectional rank operator."""
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional rank using pandas."""
        return child_result.rank(axis=1, pct=True)
```

**핵심 패턴:**
* ✅ **axis=1**: Cross-sectional operation (across columns)
* ✅ **pct=True**: Percentile ranking [0, 1]
* ✅ **NaN handling**: pandas automatically handles NaN in ranking

---

### 3.3.4. Constant Operator (`ops/constants.py`)

```python
@dataclass(eq=False)
class Constant(Expression):
    """Constant value expression."""
    value: float

    def accept(self, visitor):
        """Accept visitor for evaluation."""
        return visitor.visit_constant(self)
```

**visitor.py에서 처리:**

```python
def visit_constant(self, node: Constant) -> pd.DataFrame:
    """Create DataFrame filled with constant value."""
    # Create (T, N) DataFrame filled with constant
    result = pd.DataFrame(
        node.value,
        index=self._ctx.dates,
        columns=self._ctx.assets
    )

    # Apply universe masking
    result = result.where(self._universe_mask, np.nan)

    # Cache
    self._signal_cache[self._step_counter] = (f"Constant_{node.value}", result)
    self._step_counter += 1

    return result
```

---

## 3.4. Portfolio 구현 (`portfolio/strategies.py`)

### 3.4.1. GrossNetScaler (Unified Framework)

```python
class GrossNetScaler(WeightScaler):
    """Gross/Net exposure constraint scaler."""

    def __init__(self, target_gross: float, target_net: float):
        """Initialize with gross/net targets."""
        self.target_gross = target_gross
        self.target_net = target_net
        self.L_target = (target_gross + target_net) / 2
        self.S_target = (target_gross - target_net) / 2

    def scale(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Scale signal to portfolio weights (vectorized)."""
        # Separate positive/negative signals
        s_pos = signal.where(signal > 0, 0)
        s_neg = signal.where(signal < 0, 0)

        # Normalize (handle 0/0 → NaN → 0)
        sum_pos = s_pos.sum(axis=1)
        sum_neg = s_neg.abs().sum(axis=1)

        norm_pos = s_pos.div(sum_pos, axis=0).fillna(0)
        norm_neg = s_neg.div(sum_neg, axis=0).fillna(0)

        # Apply targets
        weights = norm_pos * self.L_target - norm_neg.abs() * self.S_target

        # Scale to meet gross target
        actual_gross = weights.abs().sum(axis=1)
        scale_factor = (self.target_gross / actual_gross).replace([np.inf, -np.inf], 0).fillna(0)
        weights = weights.mul(scale_factor, axis=0)

        return weights
```

### 3.4.2. DollarNeutralScaler (Special Case)

```python
class DollarNeutralScaler(GrossNetScaler):
    """Dollar neutral scaler (Long=1.0, Short=-1.0)."""

    def __init__(self):
        super().__init__(target_gross=2.0, target_net=0.0)
```

---

## 3.5. 사용 예시 및 Best Practices

### 3.5.1. 기본 워크플로우

```python
from alpha_database import DataSource
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank

# 1. Initialize
ds = DataSource('config')
rc = AlphaExcel(ds, start_date='2024-01-01', end_date='2024-12-31')

# 2. Define Expression
expr = TsMean(Rank(Field('returns')), window=5)

# 3. Evaluate (auto-loading, auto-masking)
result = rc.evaluate(expr)

# 4. Inspect result
print(result.head())
print(result.shape)
```

### 3.5.2. 백테스트 워크플로우

```python
from alpha_excel.portfolio import DollarNeutralScaler

# 1. Evaluate with scaler (auto-backtest)
result = rc.evaluate(expr, scaler=DollarNeutralScaler())

# 2. Access cached results
for step in range(len(rc._evaluator._signal_cache)):
    name, signal = rc._evaluator.get_cached_signal(step)
    _, weights = rc._evaluator.get_cached_weights(step)
    _, port_return = rc._evaluator.get_cached_port_return(step)

    if weights is not None:
        daily_pnl = port_return.sum(axis=1)
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        print(f"Step {step} ({name}): Sharpe = {sharpe:.2f}")
```

### 3.5.3. Serialization 워크플로우

```python
import json

# 1. Serialize Expression
expr = TsMean(Rank(Field('returns')), window=5)
expr_dict = expr.to_dict()

# 2. Save to file
with open('expression.json', 'w') as f:
    json.dump(expr_dict, f, indent=2)

# 3. Load from file
with open('expression.json', 'r') as f:
    loaded_dict = json.load(f)

# 4. Reconstruct Expression
from alpha_excel.core.expression import Expression
expr_loaded = Expression.from_dict(loaded_dict)

# 5. Extract dependencies
deps = expr_loaded.get_field_dependencies()
print(f"Required fields: {deps}")  # ['returns']
```

---

## 3.6. 테스트 전략

### 3.6.1. 단위 테스트

```python
# Test operator compute logic
def test_add_operator():
    left = pd.DataFrame([[1, 2], [3, 4]])
    right = pd.DataFrame([[5, 6], [7, 8]])

    add = Add(Constant(0), Constant(0))  # Dummy children
    result = add.compute(left, right)

    expected = pd.DataFrame([[6, 8], [10, 12]])
    pd.testing.assert_frame_equal(result, expected)
```

### 3.6.2. 통합 테스트

```python
# Test full workflow
def test_alpha_excel_workflow():
    ds = DataSource('config')
    rc = AlphaExcel(ds, start_date='2024-01-01', end_date='2024-01-31')

    expr = TsMean(Field('returns'), window=5)
    result = rc.evaluate(expr)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0  # Has rows
    assert result.shape[1] > 0  # Has columns
```

### 3.6.3. Serialization 테스트

```python
# Test round-trip
def test_serialization_round_trip():
    expr = TsMean(Rank(Field('returns')), window=5)
    expr_dict = expr.to_dict()
    expr_loaded = Expression.from_dict(expr_dict)

    # Should produce same results
    rc = AlphaExcel(ds, start_date='2024-01-01')
    result1 = rc.evaluate(expr)
    result2 = rc.evaluate(expr_loaded)

    pd.testing.assert_frame_equal(result1, result2)
```

---

## 3.7. 성능 최적화 팁

### 3.7.1. Vectorization

```python
# ✅ Good: Vectorized pandas operations
weights = signal.div(signal.abs().sum(axis=1), axis=0)

# ❌ Bad: Python loops
weights = signal.copy()
for i in range(len(signal)):
    row_sum = signal.iloc[i].abs().sum()
    weights.iloc[i] = signal.iloc[i] / row_sum
```

### 3.7.2. Caching

```python
# ✅ Good: Field auto-loading with caching
expr1 = TsMean(Field('returns'), 5)
expr2 = Rank(Field('returns'))
# 'returns' loaded only once, cached in DataContext

# ❌ Bad: Redundant loading (not possible in alpha-excel)
```

### 3.7.3. Memory Management

```python
# ✅ Good: On-demand PnL aggregation
daily_pnl = rc.get_daily_pnl(step)  # Computed when requested

# ❌ Bad: Pre-compute all aggregations
# (alpha-excel stores (T, N) port_return, aggregates on demand)
```

---

## 3.8. 향후 구현 계획

### 3.8.1. Label Quantile (HIGH PRIORITY)

**요구사항:**
- Cross-sectional quantile 기반 그룹 라벨링
- Fama-French Factor 스타일 포트폴리오 구성

**구현 계획:**

```python
# TODO: Implement in ops/crosssection.py
@dataclass(eq=False)
class LabelQuantile(Expression):
    """Cross-sectional quantile labeling for group assignment.

    Example:
        # Size factor: [Small, Big]
        size_labels = LabelQuantile(Field('market_equity'), q=2, labels=['Small', 'Big'])

        # Value factor: [Low, Medium, High]
        value_labels = LabelQuantile(Field('be_me'), q=3, labels=['Low', 'Medium', 'High'])
    """
    child: Expression
    q: int
    labels: List[str]

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional qcut with labels.

        Returns:
            DataFrame with categorical labels (T, N)
        """
        # Apply qcut row-by-row (cross-sectional)
        result = child_result.apply(
            lambda row: pd.qcut(row, q=self.q, labels=self.labels, duplicates='drop'),
            axis=1
        )
        return result
```

**사용 예시:**
```python
# Fama-French 2x3 portfolio construction
size_labels = LabelQuantile(Field('market_equity'), q=2, labels=['Small', 'Big'])
value_labels = LabelQuantile(Field('be_me'), q=3, labels=['Low', 'Medium', 'High'])

size_groups = rc.evaluate(size_labels)
value_groups = rc.evaluate(value_labels)

# Small-High portfolio
small_high_mask = (size_groups == 'Small') & (value_groups == 'High')

# Long-short strategy
long_mask = small_high_mask
short_mask = (size_groups == 'Big') & (value_groups == 'Low')
signal = long_mask.astype(float) - short_mask.astype(float)
```

---

### 3.8.2. String Universe (MEDIUM PRIORITY)

**구현 계획:**

```python
# TODO: Implement in facade.py
if isinstance(universe, str):
    # Load universe from DataSource
    universe_data = self._data_source.load_field(universe, ...)
    universe_mask = universe_data.astype(bool)
```

**사용 예시:**
```python
# Pre-defined universes in config/data.yaml
rc = AlphaExcel(ds, start_date='2024-01-01', universe='univ100')
```

---

### 3.8.3. Group Operations (MEDIUM PRIORITY)

**요구사항:**
- GroupNeutralize: 그룹별 neutralization (industry neutral 등)
- GroupDemean: 그룹별 demean

**구현 계획:**

```python
# TODO: Implement in ops/group.py
@dataclass(eq=False)
class GroupNeutralize(Expression):
    """Group-wise neutralization.

    Example:
        # Industry-neutral signal
        signal = GroupNeutralize(
            Rank(Field('returns')),
            group_by=Field('industry')
        )
    """
    child: Expression
    group_by: Expression  # Group labels (e.g., industry)

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, group_labels: pd.DataFrame) -> pd.DataFrame:
        """Apply group-wise neutralization."""
        # Implementation requires groupby logic
        pass
```

---

## 3.9. 요약

### 구현 완료 항목

✅ **Core Infrastructure**
- DataContext (dict-like pandas storage)
- AlphaExcel Facade (auto-loading, no add_data)
- EvaluateVisitor (auto-loading, double masking, triple-cache)

✅ **Expression System**
- Expression interface
- Field, Constant nodes
- Arithmetic, Logical, Time-Series, Cross-Sectional operators

✅ **Portfolio System**
- WeightScaler base class
- GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
- Shift-mask backtesting workflow

✅ **Serialization**
- Expression to/from JSON dict
- Dependency extraction

### 향후 구현 예정

🔜 **High Priority**
- Label Quantile (Fama-French Factor)

🔜 **Medium Priority**
- String Universe ('univ100', 'univ200')
- Group Operations (GroupNeutralize, GroupDemean)

---
