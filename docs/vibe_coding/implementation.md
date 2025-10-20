# 3. Implementation Guide

이 문서는 alpha-canvas의 구체적인 구현 방법론, 인터페이스 설계, 그리고 개발 표준을 정의합니다.

## 3.1. 프로젝트 구조

```text
alpha-canvas/
├── config/                      # 타입별 설정 파일
│   ├── data.yaml               # 데이터 필드 정의
│   ├── db.yaml                 # 데이터베이스 연결 설정 (선택적)
│   └── compute.yaml            # 계산 관련 설정 (선택적)
├── src/
│   └── alpha_canvas/
│       ├── __init__.py
│       ├── core/
│       │   ├── facade.py       # AlphaCanvas (rc) 퍼사드 클래스
│       │   ├── expression.py   # Expression 컴포짓 트리
│       │   ├── visitor.py      # EvaluateVisitor 패턴
│       │   └── config.py       # ConfigLoader
│       ├── ops/                # 연산자 (ts_mean, rank, etc.)
│       │   ├── __init__.py
│       │   ├── timeseries.py   # ts_mean, ts_sum, etc.
│       │   ├── crosssection.py # cs_rank, cs_quantile, etc.
│       │   └── transform.py    # group_neutralize, etc.
│       ├── analysis/
│       │   ├── pnl.py          # PnLTracer
│       │   └── metrics.py      # 성과 지표 계산
│       └── utils/
│           ├── accessor.py     # Property 접근자 (data, axis, rules)
│           └── mask.py         # 마스크 헬퍼
├── experiments/                # 실험 스크립트
├── tests/                      # 테스트
└── docs/
    └── vibe_coding/
        ├── prd.md
        ├── architecture.md
        └── implementation.md   # 이 문서
```

## 3.2. 핵심 인터페이스 설계

### 3.2.1. 초기화 및 설정

```python
from alpha_canvas import AlphaCanvas

# config/ 디렉토리의 모든 YAML 파일을 자동 로드
rc = AlphaCanvas()

# 또는 특정 config 디렉토리 지정
rc = AlphaCanvas(config_dir='./custom_config')
```

**구현 요구사항:**
- `AlphaCanvas.__init__()` 내부에서 `ConfigLoader`를 생성하고 `config/` 디렉토리의 모든 `.yaml` 파일을 로드합니다.
- `ConfigLoader`는 `data.yaml`, `db.yaml` 등을 각각 파싱하여 내부 dict에 저장합니다.

### 3.2.2. Interface A: Formula-based (Excel-like)

```python
from alpha_canvas.ops import ts_mean, rank, group_neutralize, Field

# 1. 간단한 헬퍼 함수 스타일 (즉시 평가)
returns_10d = rc.ts_mean('returns', 10)  
# rc.db['returns_10d']에 DataArray 저장

# 2. 복잡한 Expression 정의 (지연 평가)
alpha_expr = group_neutralize(
    rank(ts_mean(Field('returns'), 10)),
    group_by='subindustry'
)

# 3. Expression을 변수로 등록
rc.add_data_var('alpha1', alpha_expr)

# 4. 데이터 접근
alpha1_data = rc.data.alpha1  # xarray.DataArray (T, N)
```

**구현 요구사항:**

- `Field('returns')`: `ConfigLoader`에서 `config/data.yaml`의 `returns` 정의를 참조하는 Leaf Expression
- `ts_mean()`, `rank()` 등: Composite Expression 노드를 생성
- `rc.add_data_var()`: Expression을 `rc.rules`에 등록하고, `EvaluateVisitor`로 평가하여 `rc.db`에 저장

### 3.2.3. Interface B: Selector-based (NumPy-like)

```python
# 1. 시그널 캔버스 초기화
rc.init_signal_canvas('my_alpha')  
# rc.db['my_alpha']에 (T, N) 영행렬 생성

# 2. 데이터 등록
rc.add_data('mcap', Field('market_cap'))
rc.add_data('ret', Field('returns'))
rc.add_data('vol', Field('volume'))

# 3. 가상 축(Axis) 정의 - 레이블 기반 버킷
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=3, labels=['small', 'mid', 'big']))
rc.add_axis('momentum', cs_quantile(rc.data.ret, bins=2, labels=['low', 'high']))
rc.add_axis('surge', ts_any(rc.data.ret > 0.3, window=252))  # Boolean

# 4. 셀렉터로 Boolean 마스크 생성
mask_long = (rc.axis.size['small'] & rc.axis.momentum['high'] & rc.axis.surge)
mask_short = (rc.axis.size['big'] & rc.axis.momentum['low'])

# 5. NumPy-style 할당
rc['my_alpha'][mask_long] = 1.0
rc['my_alpha'][mask_short] = -1.0

# 또는 현재 활성 캔버스에 직접 할당
rc[mask_long] = 1.0

# 6. 최종 시그널 접근
my_alpha = rc.data.my_alpha  # xarray.DataArray (T, N)
```

**구현 요구사항:**

- `rc.add_axis()`: Expression을 `rc.rules[축이름]`에 등록
- `rc.axis.size['small']`:
  1. `rc.rules['size']` Expression 조회
  2. `Equals(expression, 'small')` 새 Expression 생성
  3. `EvaluateVisitor`로 평가하여 Boolean mask (T, N) 반환
- `rc[mask] = value`: `xr.where(mask, value, rc.db[current_canvas])`로 할당

### 3.2.4. Interface C: Selective Traceability (Integer-Based Steps)

```python
# 복잡한 Expression 정의
complex_alpha = group_neutralize(
    rank(ts_mean(Field('returns'), 5)),
    group_by='subindustry'
)

rc.add_data_var('complex_alpha', complex_alpha)

# Expression 트리 구조 (depth-first 순서):
# step 0: Field('returns')
# step 1: ts_mean(Field('returns'), 5)
# step 2: rank(...)
# step 3: group_neutralize(...)

# 1. 특정 단계만 추적
pnl_step1 = rc.trace_pnl('complex_alpha', step=1)
# {'sharpe': 0.7, 'total_pnl': 150, 'cumulative_returns': [...]}

# 2. 모든 단계 추적
pnl_all = rc.trace_pnl('complex_alpha')  # step=None (default)
# {
#   0: {'step_name': 'Field_returns', 'sharpe': 0.5, ...},
#   1: {'step_name': 'ts_mean', 'sharpe': 0.7, ...},
#   2: {'step_name': 'rank', 'sharpe': 0.6, ...},
#   3: {'step_name': 'group_neutralize', 'sharpe': 0.8, ...}
# }

# 3. 중간 데이터 직접 접근
intermediate = rc.get_intermediate('complex_alpha', step=1)
# xarray.DataArray (T, N) - ts_mean 적용 후 데이터

# 4. 복잡한 Expression 예시 (병렬 연산)
combo_alpha = ts_mean(Field('returns'), 3) + rank(Field('market_cap'))
rc.add_data_var('combo', combo_alpha)

# step 0: Field('returns')
# step 1: ts_mean(Field('returns'), 3)
# step 2: Field('market_cap')
# step 3: rank(Field('market_cap'))
# step 4: add(step1, step3)

pnl_step4 = rc.trace_pnl('combo', step=4)  # 최종 결과
```

**구현 요구사항:**

- `EvaluateVisitor.cache` 구조: `dict[str, dict[int, tuple[str, xr.DataArray]]]`
  - 외부 키: 변수명 (e.g., `'complex_alpha'`)
  - 내부 키: 정수 step 인덱스 (0부터 시작)
  - 값: `(노드_이름, DataArray)` 튜플 (디버깅용 메타데이터 포함)
- `EvaluateVisitor._step_counter`: 현재 step 인덱스를 추적하는 내부 카운터
- `EvaluateVisitor`는 Expression 트리를 **깊이 우선 탐색(depth-first)** 으로 순회하며 각 노드의 반환값을 캐시에 저장
- `rc.trace_pnl(var, step=None)`:
  - `step=None`: 모든 단계의 캐시 데이터를 `PnLTracer`에 전달
  - `step=1`: 해당 단계만 전달
- `rc.get_intermediate(var, step)`: `rc._evaluator.cache[var][step][1]` 반환 (DataArray 부분)

### 3.2.5. 핵심 활용 패턴: 팩터 수익률 계산

#### A. 독립 이중 정렬 (Independent Double Sort) - Fama-French SMB

```python
from alpha_canvas.ops import cs_quantile, Field

# 1. 데이터 등록
rc.add_data('mcap', Field('market_cap'))
rc.add_data('btm', Field('book_to_market'))

# 2. 독립 정렬: 전체 유니버스에서 각각 quantile 계산
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high']))

# 3. SMB 포트폴리오 구성
rc.init_signal_canvas('smb')
rc[rc.axis.size['small']] = 1.0   # Long all small stocks
rc[rc.axis.size['big']] = -1.0    # Short all big stocks

# 4. 팩터 수익률 추적
smb_returns = rc.trace_pnl('smb')
print(f"SMB Sharpe: {smb_returns['sharpe']:.2f}")
```

#### B. 종속 이중 정렬 (Dependent Double Sort) - Fama-French HML

```python
# 1. 첫 번째 정렬: Size
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))

# 2. 종속 정렬: 각 Size 그룹 내에서 Value quantile 계산
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high'],
                                   group_by='size'))
# group_by='size'는 rc.rules['size']를 참조하여
# 'small' 그룹과 'big' 그룹 내에서 각각 독립적으로 quantile을 계산

# 3. HML 포트폴리오 구성 (각 Size 그룹 내에서 High-Low)
rc.init_signal_canvas('hml')
rc[rc.axis.value['high']] = 1.0   # Long high B/M (value) in both size groups
rc[rc.axis.value['low']] = -1.0   # Short low B/M (growth) in both size groups

# 4. 팩터 수익률 추적
hml_returns = rc.trace_pnl('hml')
print(f"HML Sharpe: {hml_returns['sharpe']:.2f}")
```

#### C. 로우레벨 마스크 활용 (Advanced Custom Logic)

```python
# 유동성 필터링된 유니버스에서 모멘텀 팩터 구성
rc.add_data('volume', Field('volume'))
rc.add_data('returns', Field('returns'))

# 1. 고유동성 마스크 생성
high_liquidity = rc.data.volume > rc.data.volume.quantile(0.5)

# 2. 마스크 적용된 quantile 계산
rc.add_axis('momentum', cs_quantile(rc.data.returns, bins=5, labels=['q1','q2','q3','q4','q5'],
                                      mask=high_liquidity))
# mask=False인 종목은 NaN으로 처리됨

# 3. 롱-숏 포트폴리오
rc.init_signal_canvas('momentum_factor')
rc[rc.axis.momentum['q5']] = 1.0
rc[rc.axis.momentum['q1']] = -1.0

mom_returns = rc.trace_pnl('momentum_factor')
```

#### D. 다차원 팩터 조합 (Multi-Factor Strategy)

```python
# 독립 정렬로 3개 팩터 축 생성
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=3, labels=['small','mid','big']))
rc.add_axis('momentum', cs_quantile(rc.data.mom, bins=5, labels=['q1','q2','q3','q4','q5']))
rc.add_axis('quality', cs_quantile(rc.data.roe, bins=3, labels=['low','mid','high']))

# 복잡한 다차원 선택
rc.init_signal_canvas('multi_factor')

# Small & High Momentum & High Quality
long_mask = (rc.axis.size['small'] & 
             rc.axis.momentum['q5'] & 
             rc.axis.quality['high'])

# Big & Low Momentum & Low Quality
short_mask = (rc.axis.size['big'] & 
              rc.axis.momentum['q1'] & 
              rc.axis.quality['low'])

rc[long_mask] = 1.0
rc[short_mask] = -1.0

multi_returns = rc.trace_pnl('multi_factor')
```

**설계 의도:**

- 이 패턴들이 alpha-canvas의 **핵심 활용 사례**입니다.
- **Fama-French 재현**: `group_by`로 종속 정렬을 간결하게 표현
- **유연성**: `mask`로 커스텀 유니버스 필터링 가능
- 듀얼 인터페이스(Formula + Selector)의 조합으로 복잡한 다차원 팩터 포트폴리오를 간결하게 표현합니다.
- 레이블 기반 선택(`'small'`, `'q5'`, `'high'`)으로 가독성과 유지보수성을 확보합니다.

## 3.3. Cross-Sectional Quantile 연산자 구현

### 3.3.1. `cs_quantile` Expression 클래스

```python
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class CsQuantile(Expression):
    """Cross-sectional quantile bucketing with optional grouping or masking."""
    
    data: Expression  # 버킷화할 데이터 (e.g., Field('market_cap'))
    bins: int  # 버킷 개수
    labels: List[str]  # 레이블 리스트 (길이 = bins)
    group_by: Optional[str] = None  # 종속 정렬용: axis 이름 참조
    mask: Optional[Expression] = None  # 로우레벨 필터링용: Boolean Expression
    
    def accept(self, visitor: Visitor):
        return visitor.visit_cs_quantile(self)
```

### 3.3.2. `EvaluateVisitor.visit_cs_quantile` 구현

```python
def visit_cs_quantile(self, node: CsQuantile) -> xr.DataArray:
    """cs_quantile 평가: 독립/종속 정렬 및 마스크 지원."""
    
    # 1. 데이터 평가
    data = node.data.accept(self)  # (T, N) DataArray
    
    # 2. 마스크 처리 (옵션)
    if node.mask is not None:
        mask = node.mask.accept(self)  # (T, N) Boolean
        data = data.where(mask, np.nan)  # mask=False인 곳은 NaN
    
    # 3-A. 독립 정렬 (group_by=None)
    if node.group_by is None:
        return self._quantile_independent(data, node.bins, node.labels)
    
    # 3-B. 종속 정렬 (group_by 지정)
    else:
        group_expr = self._rc.rules[node.group_by]
        group_labels = group_expr.accept(self)  # (T, N) Categorical
        return self._quantile_grouped(data, group_labels, node.bins, node.labels)

def _quantile_independent(self, data: xr.DataArray, bins: int, labels: List[str]) -> xr.DataArray:
    """전체 유니버스에서 quantile 계산."""
    # xarray의 quantile 기능 활용하여 각 time step별로 cross-sectional quantile 계산
    # pd.qcut 스타일로 bins 개로 분할하고 labels 할당
    # 반환: (T, N) Categorical DataArray
    ...

def _quantile_grouped(self, data: xr.DataArray, groups: xr.DataArray, 
                      bins: int, labels: List[str]) -> xr.DataArray:
    """각 그룹 내에서 독립적으로 quantile 계산."""
    result = xr.full_like(data, fill_value='', dtype=object)
    
    for group_label in groups.unique():
        # 해당 그룹에 속하는 항목들 선택
        group_mask = (groups == group_label)
        group_data = data.where(group_mask, np.nan)
        
        # 그룹 내 quantile 계산
        group_quantiles = self._quantile_independent(group_data, bins, labels)
        
        # 결과 병합
        result = xr.where(group_mask, group_quantiles, result)
    
    return result
```

### 3.3.3. 사용 예시 비교

```python
# 독립 정렬: 간단한 Expression
size_expr = CsQuantile(
    data=Field('market_cap'),
    bins=2,
    labels=['small', 'big']
)

# 종속 정렬: group_by로 기존 axis 참조
value_expr = CsQuantile(
    data=Field('book_to_market'),
    bins=3,
    labels=['low', 'mid', 'high'],
    group_by='size'  # 'size' axis의 결과를 먼저 평가 → 각 그룹별 quantile
)

# 마스크 적용: Boolean Expression
momentum_expr = CsQuantile(
    data=Field('returns'),
    bins=5,
    labels=['q1', 'q2', 'q3', 'q4', 'q5'],
    mask=GreaterThan(Field('volume'), Quantile(Field('volume'), 0.5))
)
```

## 3.4. Property Accessor 구현

```python
# rc.data: DataAccessor
class DataAccessor:
    def __init__(self, db: xr.Dataset):
        self._db = db
    
    def __getattr__(self, name: str) -> xr.DataArray:
        if name in self._db:
            return self._db[name]
        raise AttributeError(f"Data '{name}' not found")

# rc.axis: AxisAccessor
class AxisAccessor:
    def __init__(self, rc: 'AlphaCanvas'):
        self._rc = rc
    
    def __getattr__(self, axis_name: str) -> 'AxisSelector':
        if axis_name not in self._rc.rules:
            raise AttributeError(f"Axis '{axis_name}' not defined")
        return AxisSelector(self._rc, axis_name)

class AxisSelector:
    def __getitem__(self, label: str) -> xr.DataArray:
        # rc.rules에서 Expression 조회 → Equals(..., label) 생성 → 평가
        ...
```

## 3.5. 개발 원칙

### 3.5.1. 지연 평가 (Lazy Evaluation)

- `Expression` 객체는 "레시피"이며 데이터를 가지지 않습니다.
- `EvaluateVisitor`가 실제 평가를 담당하며, 이때 캐싱이 발생합니다.
- 불필요한 재계산을 방지하여 성능을 최적화합니다.

### 3.5.2. 레이블 우선 (Label-first)

- 모든 버킷 연산은 정수 인덱스 대신 **의미 있는 레이블**을 반환해야 합니다.
- 예: `cs_quantile(..., labels=['small', 'mid', 'big'])`
- 이는 PRD의 핵심 문제 2를 해결하는 설계 원칙입니다.

### 3.5.3. 추적성 우선 (Traceability-first)

- 모든 중간 계산 결과는 **정수 step 인덱스**로 캐시되어야 합니다.
- 사용자는 재계산 없이 모든 중간 단계를 검사할 수 있어야 합니다.
- 이는 PRD의 핵심 문제 3을 해결하는 설계 원칙입니다.

### 3.5.4. Pythonic 우선 (Pythonic-first)

- 문자열 DSL 대신 Python의 네이티브 문법을 활용합니다.
- 예: `&`, `|`, `[]`, `=` 연산자 오버로딩
- IDE 자동완성 및 타입 힌트를 최대한 활용합니다.

### 3.5.5. 종속 정렬 지원 (Dependent Sort Support)

- `cs_quantile`은 `group_by` 파라미터로 종속 정렬을 지원해야 합니다.
- 이는 Fama-French 팩터 재현을 위한 핵심 요구사항입니다.
- `mask` 파라미터로 로우레벨 커스터마이징도 가능해야 합니다.

## 3.6. 테스트 전략

### 3.6.1. 단위 테스트

- `Expression` 각 노드 클래스
- `EvaluateVisitor` 메서드별 테스트 (특히 `visit_cs_quantile`의 `group_by` 로직)
- `ConfigLoader` YAML 파싱 테스트
- 정수 step 인덱싱 로직

### 3.6.2. 통합 테스트

- 전체 워크플로우 (초기화 → 데이터 로드 → Expression 평가 → PnL 추적)
- 듀얼 인터페이스 조합 시나리오
- 독립/종속 이중 정렬 기반 팩터 수익률 계산 패턴
- Fama-French SMB, HML 팩터 재현 검증

### 3.6.3. 성능 테스트

- 대용량 데이터 (T=1000, N=3000) 처리 시간
- 캐싱 효과 검증 (step별 조회 성능)
- 종속 정렬의 오버헤드 측정
- 메모리 사용량 프로파일링

## 3.7. 코딩 컨벤션

- **타입 힌트:** 모든 public 메서드에 타입 힌트 필수
- **Docstring:** Google 스타일 docstring 사용
- **Linting:** `ruff` 사용
- **Formatting:** `black` 사용
- **Import 순서:** 표준 라이브러리 → 서드파티 → 로컬

## 3.8. 인터페이스 마이그레이션 가이드

### 3.8.1. Step 인덱싱 변경사항

**이전 (문자열 기반):**
```python
# ❌ 사용하지 마세요
rc.trace_pnl('alpha1', step='ts_mean')
rc.get_intermediate('alpha1', step='ts_mean')
```

**현재 (정수 기반):**
```python
# ✅ 올바른 사용법
rc.trace_pnl('alpha1', step=1)  # step 1까지 추적
rc.get_intermediate('alpha1', step=1)  # step 1 데이터 조회

# 모든 단계 추적
rc.trace_pnl('alpha1')  # step=None (기본값)
# 반환: {0: {...}, 1: {...}, 2: {...}}
```

### 3.8.2. 독립/종속 정렬 패턴

**독립 정렬 (변경 없음):**
```python
# ✅ 기존과 동일하게 작동
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small','big']))
```

**종속 정렬 (신규 기능):**
```python
# ✅ 새로운 기능
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low','mid','high'],
                                   group_by='size'))
```

**마스크 기반 필터링 (신규 기능):**
```python
# ✅ 새로운 기능
mask = rc.data.volume > threshold
rc.add_axis('filtered', cs_quantile(rc.data.returns, bins=5, labels=[...],
                                      mask=mask))
```

## 3.9. 구현 성공 기준

### 3.9.1. Step 인덱싱 검증

✅ **필수 동작:**
- `rc.trace_pnl('alpha', step=2)` → step 2까지의 PnL 반환
- `rc.get_intermediate('alpha', step=2)` → step 2의 캐시된 DataArray 반환
- 병렬 Expression (브랜치가 있는 트리)에서 올바른 순서로 인덱싱
- 잘못된 step 인덱스 입력 시 명확한 에러 메시지

### 3.9.2. 종속 정렬 검증

✅ **필수 동작:**
- **독립 정렬**: `cs_quantile(...)` → 전체 유니버스 대상 quantile
- **종속 정렬**: `cs_quantile(..., group_by='axis')` → 각 그룹 내 quantile
- **마스크 필터링**: `cs_quantile(..., mask=...)` → 필터링된 부분집합 대상 quantile
- 독립/종속 정렬의 결과 cutoff가 명확히 다름 (검증 테스트 필요)

### 3.9.3. Fama-French 재현 검증

✅ **필수 동작:**
- SMB (독립 2×3 정렬) → 예상된 포트폴리오 가중치 생성
- HML (종속 2×3 정렬) → 예상된 포트폴리오 가중치 생성
- 독립/종속 방식의 cutoff 차이 검증 (academic paper 기준과 일치)

## 3.10. 다음 단계

### Phase 1: 핵심 컴포넌트 구현
- [ ] `Expression` 추상 클래스 및 Leaf/Composite 구현
- [ ] `EvaluateVisitor` 기본 구조 및 캐싱 메커니즘 (정수 step 카운터 포함)
- [ ] `ConfigLoader` 및 YAML 파싱
- [ ] `AlphaCanvas` Facade 기본 구조

### Phase 2: 연산자 구현
- [ ] Timeseries 연산자 (`ts_mean`, `ts_sum`, etc.)
- [ ] Cross-sectional 연산자 (`cs_rank`, `cs_quantile` with `group_by` and `mask`)
- [ ] Transform 연산자 (`group_neutralize`, etc.)

### Phase 3: 추적성 및 분석
- [ ] `PnLTracer` 구현
- [ ] 선택적 단계 추적 로직 (정수 인덱스 기반)
- [ ] 성과 지표 계산
- [ ] PnL 리포트에 step 메타데이터 표시

### Phase 4: 인터페이스 완성
- [ ] Property accessor (`rc.data`, `rc.axis`)
- [ ] NumPy-style 할당 (`rc[mask] = value`)
- [ ] 헬퍼 메서드 (`rc.ts_mean()` 등)

### Phase 5: 검증 및 테스트
- [ ] 정수 step 인덱싱 단위 테스트
- [ ] `_quantile_grouped` 로직 단위 테스트
- [ ] Fama-French SMB/HML 통합 테스트
- [ ] 종속 정렬 성능 벤치마크

---

**참고:** 이 문서는 실제 구현 과정에서 발견되는 새로운 패턴과 교훈을 지속적으로 반영해야 합니다 (Living Document).

