# 1\. 제품 요구사항 문서 (Product Requirement Document)

## 1.1. 개요

* **제품명:** alpha-canvas
* **목표:** 퀀트 리서처가 Python 네이티브 환경에서 복잡한 알파 아이디어를 빠르고 직관적으로 테스트하고, 그 과정을 투명하게 추적할 수 있는 차세대 리서치 플랫폼을 제공합니다.
* **비전:** 데이터 검색부터 팩터 생성, PnL 분석까지의 전 과정을 통합된 단일 인터페이스(`rc`)로 제공하여, 팩터 수익률(factor return) 계산을 포함한 아이디어의 프로토타이핑 속도를 획기적으로 단축시킵니다.

## 1.2. 핵심 문제 (The Problem)

기존 퀀트 리서치 툴(e.g., WorldQuant BRAIN)은 강력하지만 다음과 같은 명확한 한계를 가집니다.

1. **복잡하고 비직관적인 문법:** `bucket(rank(x), range="0.1, 1, 0.1")`와 같이 문자열 파싱에 의존하는 복잡한 문법은 배우기 어렵고 오류를 유발하기 쉽습니다.
2. **조작 불가능한 "블랙박스" 버킷:**
      * 기존 인터페이스는 버킷(bucket) 연산의 결과로 `'small'`, `'mid'`와 같은 의미 있는 \*\*레이블(Label)\*\*이 아닌, `0, 1, 2` 같은 \*\*정수 인덱스(Index)\*\*를 반환합니다.
      * 이로 인해 리서처가 "중간(mid) 그룹은 제외하고 싶다" 또는 "Small & High 그룹만 골라내고 싶다"와 같이 특정 그룹을 \*\*선택(Select)\*\*하여 데이터를 조작하는 것이 사실상 불가능합니다.
3. **결과 중심의 불투명성 (낮은 추적성):** `group_neutralize(ts_mean('returns', 3), ...)`와 같은 복잡한 수식의 최종 PnL만 알 수 있을 뿐, *중간 단계*(`ts_mean` 적용 직후)에서 `NaN`이 발생했는지, 또는 어느 연산이 PnL을 하락시켰는지 추적하기 매우 어렵습니다.

## 1.3. 대상 사용자 (User Persona)

* **페르소나:** 바이사이드(Buy-side) 퀀트 리서처 / 포트폴리오 매니저
* **특징:**
  * Python 및 `pandas`/`numpy`/`xarray`에 익숙합니다.
  * 아이디어를 빠르게 프로토타이핑하고 싶어 합니다.
  * 단순히 PnL 결과만 보는 것이 아니라, 수식의 **각 단계별**로 PnL 기여도와 데이터 상태를 투명하게 추적할 수 있기를 원합니다.

## 1.4. 주요 기능 요구사항 (Key Features)

### F1: Config 기반 데이터 검색 (Data Retrieval)

* **요구사항:** 사용자는 `config/` 디렉토리 내의 타입별 YAML 파일(e.g., `data.yaml`, `db.yaml`)에 설정을 명시적으로 정의할 수 있어야 합니다. 이는 데이터 로직과 리서치 로직을 명확히 분리합니다.
* **Config 구조:**
  * `config/data.yaml`: 데이터 필드(field) 별칭 및 DB 접근 정보
  * `config/db.yaml`: 데이터베이스 연결 설정 (선택적)
  * `config/calc.yaml`: 기타 계산 관련 설정 (선택적)
* **Config 예시 (`config/data.yaml`):**

```yaml
# 'adj_close'라는 별칭(alias) 정의
adj_close:
    table: PRICEVOLUME  # 실제 DB 테이블명
    index_col: date
    security_col: securities
    value_col: adj_close
    # SQL 쿼리를 직접 사용하여 (T, N) 데이터를 가져옴
    query: >
    SELECT 
        TRD_DD as date, 
        TICKER as securities, 
        CLOSE * ADJ_FACTOR as adj_close 
    FROM PRICEVOLUME

# 'subindustry'라는 별칭 정의 (그룹핑/메타데이터용)
subindustry:
    table: SECURITY_MASTER
    index_col: date
    security_col: securities
    value_col: GICS_SUBINDUSTRY
```

### F2: 듀얼 인터페이스 (Dual Interface)

사용자는 리서치 목적에 따라 두 가지 방식의 인터페이스를 자유롭게 혼용할 수 있어야 합니다.

* **인터페이스 A: Excel-like Formula (수식 기반)**

  * **요구사항:** `ts_mean`, `rank` 등 WQ 스타일의 연산자를 단순하고 파이썬다운 함수 호출로 지원해야 합니다.
  * **시나리오:**

```python
# 1. 헬퍼 함수로 (T, N) DataArray 즉시 받기
returns_10d = rc.ts_mean('return', 10) 

# 2. 복잡한 룰(Expression)을 정의하고 'alpha1' 변수로 저장
alpha_expr = group_neutralize(ts_mean('return', 10), 'subindustry')
rc.add_data_var('alpha1', alpha_expr) 
```

* **인터페이스 B: 셀렉터 인터페이스 (Numpy-style)**

  * **요구사항:** (핵심 문제 2 해결) "가상의 축"(Axis 룰)을 동적으로 정의하고, `numpy`처럼 불리언 마스크를 생성하여 최종 시그널 캔버스에 값을 할당할 수 있어야 합니다.
  * **시나리오:**

```python
# 1. "시그널 캔버스" 초기화 (rc.db에 'my_alpha' DataArray 생성)
rc.init_signal_canvas('my_alpha') 

# 2. "데이터 룰" 등록 (필요시)
rc.add_data('mcap', Field('market_cap'))
rc.add_data('ret', Field('return'))

# 3. "가상 축(Axis) 룰" 동적 등록
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))
rc.add_axis('surge_event', ts_any(rc.data.ret > 0.3, window=504))

# 4. "셀렉터"로 (T, N) 불리언 마스크 생성 
mask_long = (rc.axis.size['small'] & rc.axis.surge_event)

# 5. "Numpy-style 할당" (rc[mask] = 1.0)
rc[mask_long] = 1.0  
```

### F3: 심층 추적성 (Deep Traceability)

* **요구사항:** (핵심 문제 3 해결) 사용자는 복잡한 수식의 **각 중간 연산 단계별**로 `(T, N)` 데이터 상태와 PnL을 **선택적으로** 추적할 수 있어야 합니다. 단계는 **정수 인덱스(0부터 시작)**로 참조되며, Expression 트리의 **깊이 우선 탐색(depth-first) 순서**를 따릅니다.

* **단계 인덱싱 예시:**
  * Expression: `group_neutralize(ts_mean(Field('returns'), 3), 'subindustry')`
  * **step 0**: `Field('returns')` → 원본 returns 데이터
  * **step 1**: `ts_mean(Field('returns'), 3)` → 이동평균 적용 후
  * **step 2**: `group_neutralize(...)` → 그룹 중립화 후 (최종)

* **시나리오 1: 특정 단계 추적**
    1. 사용자가 `alpha_expr = group_neutralize(ts_mean(Field('returns'), 3), 'subindustry')` 룰을 정의합니다.
    2. `rc.add_data_var('alpha1', alpha_expr)`를 실행합니다.
    3. 이후 `rc.trace_pnl('alpha1', step=1)`를 호출합니다.
    4. 시스템은 **재계산 없이** step 1 (`ts_mean` 적용 후)의 중간 결과에 대한 PnL만 반환합니다.

* **시나리오 2: 전체 단계 추적**
    1. 사용자가 `rc.trace_pnl('alpha1', step=None)`를 호출합니다. (또는 `step` 생략)
    2. 시스템은 모든 중간 단계의 PnL 리포트를 순서대로 반환합니다:
          * **step 0**: 원본 `returns`의 PnL
          * **step 1**: `ts_mean` 적용 후의 PnL
          * **step 2**: `group_neutralize` 적용 후 최종 PnL

* **시나리오 3: 중간 데이터 접근**
    1. 사용자가 `rc.get_intermediate('alpha1', step=1)`를 호출합니다.
    2. 시스템은 step 1 (`ts_mean` 적용 후)의 `(T, N)` `DataArray`를 반환하여 직접 검사할 수 있게 합니다.

* **복잡한 Expression 예시:**
  * Expression: `ts_mean(Field('returns'), 3) + rank(Field('market_cap'))`
  * **step 0**: `Field('returns')` → 원본 returns
  * **step 1**: `ts_mean(Field('returns'), 3)` → returns의 이동평균
  * **step 2**: `Field('market_cap')` → 원본 market cap
  * **step 3**: `rank(Field('market_cap'))` → market cap의 순위
  * **step 4**: `add(step1, step3)` → 최종 합계

### F4: 팩터 수익률 계산 (Core Use Case)

* **요구사항:** F2의 듀얼 인터페이스를 활용하여 **다차원 팩터 포트폴리오 구성 및 수익률 계산**을 간결하게 수행할 수 있어야 합니다. 이는 alpha-canvas의 핵심 활용 사례입니다. 특히 **Fama-French 3-factor 모델**의 독립/종속 이중 정렬(independent/dependent double sort)을 쉽게 재현할 수 있어야 합니다.

* **시나리오 1: 독립 이중 정렬 (Independent Double Sort)**
    1. 사용자가 두 개의 팩터(Size, Value)를 정의합니다.
    2. 각 팩터를 **독립적으로** 버킷화(quantile)하여 축(axis)으로 등록합니다.
    3. 셀렉터 인터페이스를 사용하여 다차원 교차 선택을 수행합니다.
    4. 예시: Fama-French SMB (Small Minus Big) 팩터

```python
# 1. 독립 정렬: 전체 유니버스에서 각각 quantile 계산
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high']))

# 2. Small-Value 포트폴리오 구성
rc.init_signal_canvas('smb')
rc[rc.axis.size['small']] = 1.0   # Long small stocks
rc[rc.axis.size['big']] = -1.0    # Short big stocks

# 3. 팩터 수익률 계산
smb_returns = rc.trace_pnl('smb')
```

* **시나리오 2: 종속 이중 정렬 (Dependent Double Sort)**
    1. 첫 번째 팩터(Size)로 먼저 정렬합니다.
    2. 두 번째 팩터(Value)는 **첫 번째 팩터의 각 그룹 내에서** 독립적으로 quantile을 계산합니다.
    3. 이를 위해 `group_by` 파라미터를 사용합니다.
    4. 예시: 종속 정렬 기반 HML (High Minus Low) 팩터

```python
# 1. 첫 번째 정렬: Size
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))

# 2. 종속 정렬: 각 Size 그룹 내에서 Value quantile 계산
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high'],
                                   group_by='size'))

# 3. HML 포트폴리오 구성 (각 Size 그룹 내에서 High-Low)
rc.init_signal_canvas('hml')
rc[rc.axis.value['high']] = 1.0   # Long high B/M (value stocks)
rc[rc.axis.value['low']] = -1.0   # Short low B/M (growth stocks)

# 4. 팩터 수익률 계산
hml_returns = rc.trace_pnl('hml')
```

* **시나리오 3: 로우레벨 마스크 기반 정렬 (고급)**
    1. `group_by`로 표현하기 어려운 커스텀 조건부 정렬이 필요한 경우
    2. `mask` 파라미터로 특정 조건을 만족하는 종목들에 대해서만 quantile 계산

```python
# 유동성이 충분한 종목들만 대상으로 모멘텀 quantile 계산
high_liquidity_mask = rc.data.volume > rc.data.volume.quantile(0.5)
rc.add_axis('momentum', cs_quantile(rc.data.returns, bins=5, labels=['q1','q2','q3','q4','q5'],
                                      mask=high_liquidity_mask))
```

-----
