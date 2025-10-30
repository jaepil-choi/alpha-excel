# Alpha Canvas v1.0 --> v2.0 전환 계획

## 1.0의 문제점들

- Visitor 패턴의 Lazy Execution 구조 때문에 느림.
- 연산자를 일일이 import 해야 함.

```python
from alpha_excel.ops.constants import *
from alpha_excel.ops.arithmetic import *
from alpha_excel.ops.timeseries import *
from alpha_excel.ops.group import *
from alpha_excel.ops.crosssection import *
from alpha_excel.ops.transformation import *
# bad
```

- 모든 signal, portfolio weights, portfolio returns를 cache에 저장해 메모리 사용량이 크고, 속도가 느림.
- 불필요하게 DataSource를 따로 불러와야 함.

```python
# DataSource 인스턴스 생성
ds = DataSource()

# AlphaExcel 인스턴스 생성
ae = AlphaExcel(
    data_source=ds, # bad
    start_date="2020-01-01",
    end_date="2024-12-31",
)
```

- 마지막에 step index로 cache에 접근해야 하는 것은 매우 불편함.
  - 저장된 step들을 쉽게 볼 수 있어야 하고,
  - step의 expression이 나와야 하며,
  - step에 접근할 때 Scaler로 portfolio weight 및 returns를 계산해야 함. (미리 계산해놓지 않아야 함.)

## 내가 생각한 해결책

## 전반적인 구조 변경

- visitor 패턴을 버리고, eager execution 구조로 바꿔야 함.
- Field로 데이터를 불러올 때 data의 type이 같이 불러와져야 함.
  - 예: group, numeric, object / weight, port_return 등
  - data의 type은 data.yaml에 명시되어 있어야 함.
  - group일 경우 불러올 때 기본적으로 fully ffill 하지만 data.yaml에서 ffill: 0 으로 override 할 수 있음.
    - ffill: -1이면 fully ffill
    - 다른 데이터들은 기본적으로 ffill: 0
    - numeric 데이터라도 ffill: 5와 같이 override 가능.
- Operator들이 output을 return하기 전 output data model의 data type를 지정해야 함.
- operator들은 input을 위치에 따라 data type을 체크해야 함.
  - 예를들어 group_neutralize(A, B)의 경우 (A:numeric, B:group) 타입이어야 함.
- data model과 operator의 구조를 바꿔야 함.
  - operator들은 stateless하지만, output인 data model은 stateful해야 함.
  - 우선 field로 불러온 데이터는 항상 `step_counter=0`임.
    - operator를 적용할 때마다 output data model의 step counter가 1씩 증가함.
  - output data model은 step history를 가지고 있어야 함. (각 step마다 expression과 파라미터를 저장)
    - step history는 operator를 적용할 때마다 자동으로 추가됨.
      - nested 방식으로 step마다 expression과 파라미터를 기록
    - data model의 __repr__은 지금까지 적용된 expression을 str으로 보여줘야 함.
      - 예: group_neutralize(ts_mean(returns, window=5), fn_industry_group)
  - operator에 공통 파라미터 `record_output=True`를 하면 output data model에 data도 저장됨.
    - 즉, 유저가 원하는 데이터만 cache할 수 있음.
    - 이렇게 data가 cache 된 data model이 다음 operator의 input으로 들어온 경우, output으로 나가는 data model에도 data cache가 유지되어야 함. 
- data model은 여전히 +, -, *, /, ** (arithmetic) 연산자를 오버라이드 해야 함.
  - 이 경우에도 output data model의 step counter가 1씩 증가하고, step history에 해당 연산이 기록되어야 함.
- universe masking 기능
  - Expression의 하위 클래스인 Field와 Operator 모두 output을 return하기 전에 ae = AlphaExcel() 인스턴스를 initialize 했을 때 생성된 universe로 마스킹 시켜야 함.
  - field에 데이터 불러와 output으로 내보내기 전에 universe masking
  - operator 적용 후 output 내보내기 전에 universe masking
    - 예를들어 ts_mean(numeric_field, window=5) 의 output은 numeric type임.
    - group_neutralize(numeric_field, group_field) 의 output도 numeric type임.
    - 하지만 concat_group(group_field1, group_field2)의 output은 group type임.

### 인터페이스

- DataSource 인스턴스를 AlphaExcel 인스턴스에 직접 넘기지 말고, AlphaExcel 인스턴스가 내부적으로 DataSource 인스턴스를 생성하게 해야 함.
- operator나 field 모두 pandas DataFrame/Numpy 2d array를 반환해야 함. (output.to_df 또는 output.to_numpy() 메서드 제공)
- 오퍼레이터들은 o.ts_mean()와 같은 방법으로 모두 접근할 수 있어야 하고, 필드는 f('returns') 처럼 접근할 수 있어야 함.
  - 오퍼레이터들은 클래스이기 때문에 기본적으로 대문자로 시작하는 Camel Casing이지만, 소문자 메서드 alias가 있어야 함. (예: TsMean -> ts_mean)
- group 오퍼레이터들은 group_by=str 방식으로 파라미터를 받지 않음.
  - group_neutralize(field, group_field) 처럼 받아야 함.

### Portfolio weight와 returns 계산

- Scaler도 AlphaExcel 인스턴스 facade에서 접근할 수 있어야 함.
  - list scalers
  - set scaler
- Portfolio Weight (scaler 적용)
  - 임의의 data model을 input으로 받아 stateless하게 portfolio weight가 저장된 data model을 return해줘야 함.
    - 예를들어 `ae.to_weights(data_model)` 같은 메서드로 접근.
    - output data model은 weight type이어야 함.
- Portfolio Returns
  - 임의의 portfolio weight data model을 input으로 받아 stateless하게 portfolio returns가 저장된 data model을 return해줘야 함.
    - 예를들어 `ae.to_portfolio_returns(weight_data_model)` 같은 메서드로 접근.
    - weight_data.shift(1) 후 universe mask를 적용한 후 returns와 곱해서 portfolio returns를 계산.
    - output data model은 port_return type이어야 함.
  - to_long_returns, to_short_return 메서드도 제공.
    - 각각 long 포지션들의 returns만, short 포지션들의 returns만 계산.

### Operator와 Field의 효율적 연산

- 어떤 operation의 경우 pandas로 처리하는 것이, 어떤 경우 numpy로 처리하는 것이 더 빠르거나 간결할 수 있음.
  - operator는 data model의 input이 들어오면 data_model.to_df()를 쓸 것인지, data_model.to_numpy()를 쓸 것인지 결정해야 함. (클래스의 속성으로 미리 정해놓기)
  - 예를들어 ts_mean의 경우 pandas로 처리하는 것이 더 간결함. (rolling.mean())
- group 데이터의 경우 불러오면 string이 아닌 category type으로 불러와야 함.
- group 연산의 경우 어차피 이중 반복이 불가피함
  - dataframe을 row by row iterate하는 것은 느림.
  - group 연산자들이 공통으로 쓸 수 있는 **어떤 방법**이 필요함.

### 기타

- operator을 만들 때, 기존 operator를 사용하는 것이 가능해야 하며, 권장됨.
  - 예: ts_zscore = (field - ts_mean(field)) / ts_std(field)
