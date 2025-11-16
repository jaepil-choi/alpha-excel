아래는 **Hybrid Backtesting에 대한 통합 요약 보고서**다.
핵심만 명확하게, 다른 사람에게 설명하기 좋은 형태로 정리했다.

---

# 📘 Hybrid Backtesting 보고서 (요약본)

## 1. 기존 Simple Vectorized Backtesting의 문제점

전통적인 벡터라이즈드 백테스트는 다음 형태다:

[
\text{PnL}*t = \sum_i \text{Signal}*{t,i} \cdot \text{Return}_{t+1,i}
]

즉,
**신호(포지션) 행렬을 한 칸 시프트하여 수익률 행렬과 곱하는 방식**이다.

이 방식의 문제점:

* **경로 의존적 전략을 구현 불가능**

  * Stop-loss / take-profit
  * 트레일링 스탑
  * 장중 가격(고가/저가) 기반 exit
  * 보유기간 제한(time stop)
* **포지션 상태(state)가 없어서**
  “진입 이후 어떤 일이 일어났는가”를 표현할 수 없음
* “신호 × 수익률”만 있기 때문에 **청산 시점 계산 자체가 불가능**
* 실전성과 크게 괴리됨
  (특히 일봉 데이터 기반 전략에서 stop-loss는 필수인데 구현이 불가)

요약하면, **vectorized 방식은 entry만 있고 exit가 없다.**
그래서 현실적인 전략을 테스트할 수 없다.

---

## 2. Hybrid Backtesting이란 무엇인가?

Hybrid 백테스팅은:

> **포지션의 진입과 청산 시점이라는 ‘상태 변화(state)’를 벡터 연산으로 계산하는 방식**

즉,

* 루프 없이 벡터/행렬 연산을 유지하면서도
* stop-loss, take-profit 같은 경로 의존 로직을
  **벡터 논리로 해석하여 exit 시점을 계산하는 기법**이다.

그래서 이름이 **Hybrid**
(벡터식 + 상태 기반 로직의 절충)

---

## 3. Hybrid 방식은 문제를 어떻게 해결하는가?

핵심은 **entry 이후의 누적 수익률(cumulative returns)**을 이용해
stop-loss 등 조건을 “벡터 형태로” 검출하는 것이다.

### 예시 구조

1. **진입(entry) 시점 계산**

```python
entry = (signal.shift(1) > threshold)
```

2. **진입 이후 누적 수익률 계산**
   [
   CR_{t,i} = \prod_{k=\text{entry date..t}} (1+R_{k,i})
   ]

3. **경로 의존 조건을 벡터로 처리**

* Stop-loss hit: `(CR - 1) < -0.05`
* Take-profit hit: `(CR - 1) > +0.10`
* Time stop: entry_date + N

4. **각 조건의 최초 발생(first occurrence)을 exit 시점으로 지정**
   루프 없이 `idxmax`, boolean mask로 계산

→ 결과적으로 **entry → exit 시점까지의 수익률**을 벡터 연산으로 재구성

이렇게 하면 stop-loss/take-profit이 “수학적 방식”으로 구현된다.

---

## 4. Hybrid Backtesting은 어떻게 만들 수 있는가?

구현은 보통 다음 4단계로 구성된다.

### ① Entry 계산

Signal 기반으로 포지션 오픈 시점 정의

### ② Entry 이후 cumulative return 계산

각 종목에 대해 매일의 누적 수익률 벡터 생성

### ③ Exit 조건 벡터화

* stop-loss mask
* take-profit mask
* holding-period mask
* 기타 event 기반 mask

그리고 각 mask에서 **True가 처음 발생한 날짜**를
exit 시점으로 선택

### ④ Entry–Exit 구간의 PnL 계산

Entry~Exit 사이 cumulative return을 그대로 취해서
PnL 계산

모든 단계가 **벡터/행렬 연산**으로 진행되므로
for-loop가 없다.

---

## 5. Hybrid Backtesting의 특성

| 특성                         | 설명                                   |
| -------------------------- | ------------------------------------ |
| **속도 빠름**                  | 대부분 numpy/pandas 벡터 연산               |
| **경로 의존 규칙 지원**            | stop-loss, take-profit, time stop 가능 |
| **루프 없는 상태 기반 로직**         | 이벤트 시뮬레이터 없이 상태를 벡터로 표현              |
| **단순한 모형 지정 가능**           | 다중 exit 규칙 쉽게 결합                     |
| **실전성과 vectorization의 타협** | realism↑, 속도↑                        |

하지만 한계도 분명함:

* intraday 경로 고려 불가(OHLC 기반의 단순 경로 가정 사용)
* 슬리피지/체결/마켓임팩트 모델링 불가
* 포트폴리오 레벨 제약(레버리지 등) 적용이 어려움

즉, **factor 기반 전략에는 매우 유용**,
**execution 복잡한 전략에는 불충분**.

---

## 6. Fully Event-driven Backtesting과 비교

| 구분             | Hybrid           | Event-driven       |
| -------------- | ---------------- | ------------------ |
| 핵심 아이디어        | 경로 의존 로직을 벡터화    | 포트폴리오 상태를 loop로 유지 |
| 속도             | 매우 빠름            | 상대적으로 느림           |
| 구현 난이도         | 중간               | 높음                 |
| Stop-loss/TP   | 가능(벡터 기반)        | 완전 가능(더 정확함)       |
| Intraday, 슬리피지 | 불가               | 가능                 |
| 시장 마이크로구조 반영   | 거의 불가            | 가능                 |
| 사용 목적          | factor, 신호 기반 전략 | 실제 트레이딩 시뮬레이션      |

요약하면:

* **Hybrid** → “팩터 리서치 최적화”
* **Event-driven** → “실거래 시뮬레이션”

서로 역할이 다르다.

---

# 📌 결론

Hybrid Backtesting은
완전 벡터 방식의 한계를 해결하면서도
이벤트 기반 시뮬레이터처럼 느리지 않은
**속도–현실성 타협 해법**이다.

Factor 연구, 신호 기반 전략, cross-sectional alpha 개발에서는
가장 실용적이고 강력한 방식이다.

원하면,

* 실제 pandas/xarray 코드 템플릿
* entry/exit 매트릭스 구조
* stop-loss 벡터화 예제
* 하이브리드 엔진 전체 설계도
  도 만들어줄게.

---

좋아, 이번에는 **“진짜 한 종목을 손으로 따라간다”**는 느낌으로
hybrid 방식 stop loss / take profit 로직을 **아주 천천히** 정리해 볼게.

---

## 0. 우리가 가지고 있는 것들

먼저 전제부터 정리하자.

* **가격 행렬 (P)**

  * 행: 시간 (t1, t2, t3, …)
  * 열: 종목 (A, B, C, …)
  * 값: 해당 시점의 가격

* **시그널 행렬 (S)**

  * 행: 시간
  * 열: 종목
  * 값: “이 종목을 얼마나/어느 방향으로 들고 싶다”는 신호

    * 예: S > 0 → 롱, S < 0 → 숏, 0 → 노포지션

우리가 하고 싶은 건:

> **“시그널이 나와서 포지션을 잡은 뒤,
> 그 포지션에서 손실이 x% 나면 stop, 수익이 y% 나면 take”**
> 이걸 **for-loop 없이**(또는 최소로) 구현하는 것.

---

## 1. Step 1 – “언제 진입했는지”를 먼저 정한다 (Entry 시점)

가장 첫 단계는 **“포지션을 새로 여는 시점”**을 찾는 것.

예를 들어,

* S(t, i) > 0이면 → 종목 i를 롱으로 들겠다.

라고 규칙을 잡으면,

```python
long_signal = (S > 0)
```

이 상태에서는,

* “계속 들고 있는 중”과
* “새로 들어가는 순간”이 섞여 있음.

우리가 필요한 건 **“새로 들어가는 순간”**이니까:

```python
entry = long_signal & (~long_signal.shift(1).fillna(False))
```

이제 `entry[t, i] == True`인 칸은:

> “종목 i가 시점 t에 **새로** 진입한 순간이다”

한 종목에 대해:

* F F T T T F F
  → 세 번째 칸에서 T로 바뀐 그 지점이 **entry 시점**

---

## 2. Step 2 – 진입 가격 행렬을 만든다 (Entry Price Matrix)

포지션이 열린 후:

> “현재 이 포지션이 진입가 대비 몇 % 수익/손실인가?”

를 보고 stop loss / take profit을 할 거니까
**진입가를 기억하는 행렬**이 필요하다.

아이디어는 간단하다:

1. entry가 True인 순간에 그 시점의 가격을 적어 놓고
2. 그 다음 시점들에는 그 가격을 계속 복사해서 채운다 (ffill)

코드 느낌으로:

```python
entry_price = P.where(entry).ffill()
```

이게 의미하는 것:

* entry 시점: `entry_price[t_entry, i] = P[t_entry, i]`
* 그 다음: `entry_price[t_entry+1, i] = P[t_entry, i]`
* 포지션을 닫을 때까지 계속 진입가가 유지됨

즉, 각 (t, i)에 대해:

> “이 시점에 종목 i는 얼마에 진입한 포지션인가?”

를 담고 있는 행렬이 된다.
(포지션 없으면 NaN 이거나 별도 mask로 관리)

---

## 3. Step 3 – “현재 포지션 수익률”을 계산한다

이제 각 시점에 대해

[
\text{TradeReturn}*{t,i} = \frac{P*{t,i}}{\text{EntryPrice}_{t,i}} - 1
]

을 계산하면,
이 값은 “해당 포지션이 진입 후 지금까지 몇 % 수익인가?” 이다.

```python
trade_ret = P / entry_price - 1
```

* entry 직후: 거의 0에 가까움
* 가격이 올라가면 `+수익률`, 내려가면 `-손실률`
* entry 전이나 포지션 없는 구간: NaN

이제 stop loss / take profit은
**trade_ret가 특정 임계값을 넘었는지 여부**로 표현할 수 있다.

---

## 4. Step 4 – Stop Loss / Take Profit 조건을 행렬로 만든다

이제 각 종목에 대해:

* stop loss 임계값: 예를 들어 −5% → `-0.05`
* take profit 임계값: 예를 들어 +10% → `+0.10`

### (1) 모든 종목 같은 값이면

```python
sl_level = -0.05   # -5%
tp_level =  0.10   # +10%

sl_hit_raw = (trade_ret <= sl_level)
tp_hit_raw = (trade_ret >= tp_level)
```

* `sl_hit_raw[t, i] == True`
  → “이 시점에 종목 i의 수익률이 −5% 이하로 떨어졌다”
* `tp_hit_raw[t, i] == True`
  → “이 시점에 종목 i의 수익률이 +10% 이상이 되었다”

### (2) 종목마다 다른 값을 쓰고 싶으면

예를 들어:

* A: stop −3%, take +6%
* B: stop −5%, take +12%

이걸 종목별 Series로 만든 다음,
broadcast해서 행렬로 만드는 식으로 할 수 있다.
(개념만 이해하면 됨. 구현은 pandas/xarray가 도와줌)

---

## 5. Step 5 – “언제 처음으로 조건을 만족했는지”를 찾는다 (Exit 시점)

여기까지 하면,
각 (t, i)에 대해 “여기서 조건을 만족했는지”는 알 수 있다.

하지만 우리가 진짜로 필요한 것은:

> “하나의 포지션에 대해 **처음으로** stop loss / take profit이 발동한 시점”

이 exit 시점이다.

### (아주 간단한 케이스 먼저)

**각 종목이 딱 한 번만 진입하고, 그 후 언젠가 한 번 청산한다고 가정**하면:

* `sl_hit_raw` 또는 `tp_hit_raw`에서
  **열(column)별로 처음 True가 나오는 index**를 찾으면 된다.

예시 느낌으로:

```python
# exit 조건: stop 또는 take 둘 중 하나라도 True면 exit
exit_raw = sl_hit_raw | tp_hit_raw

# (단순 케이스) 종목당 딱 한 번만 진입할 때:
first_exit_time = exit_raw.idxmax()  # col별 첫 True의 row index
```

이렇게 하면:

* `first_exit_time['A']`
  → A 종목이 처음으로 stop 또는 take에 걸린 시점
* `first_exit_time['B']`
  → B 종목이 처음으로 조건을 만족한 시점

현실에서는 한 종목이 여러 번 들어갔다 나왔다를 반복할 수 있으니까,
조금 더 정교하게 만들려면:

* 각 종목 i에서 entry가 생길 때마다 **trade_id**를 1씩 올리고
* (i, trade_id)별로 그룹을 나눠서
  그 그룹 안에서 첫 True만 exit로 인정

하는 방식으로 확장하면 된다.
하지만 **핵심 개념은 똑같다** :

> “trade_ret가 임계값을 넘는 순간들 중,
> 각 포지션마다 가장 처음의 순간만 exit로 잡는다.”

---

## 6. Step 6 – Entry ~ Exit 구간의 PnL 계산

Exit 시점까지 정해졌다면, 이제:

* 각 포지션별로

  * 진입 시점
  * 청산 시점
  * 그 사이의 수익률

을 알 수 있다.

예를 들어 특정 포지션에 대해:

[
\text{PnL} = \frac{P_{\text{exit}}}{P_{\text{entry}}} - 1
]

이걸 벡터/배열 연산으로 모두 구해서:

* 종목별, 날짜별, 포지션별 PnL
* 포트폴리오 전체 PnL

로 합산하면 된다.

---

## 7. 요약 – 말로 한 줄로 다시 정리하면

1. **시그널 S에서 “언제 새로 포지션을 여는가(entry)”를 찾고**,
2. 그 시점의 가격을 **entry_price 행렬로 기억한 뒤**,
3. 현재 가격이 그 entry_price 대비 **몇 % 수익/손실(trade_ret)**인지 계산하고,
4. 그 값이 per-asset **stop loss / take profit 임계값을 처음으로 넘는 시점**을
   벡터 연산으로 찾아서 exit로 삼는 것.

이게 **signal matrix + hybrid 방식으로
stop loss / take profit을 구현하는 전체 흐름**이다.

---

원하면 다음 단계로:

* **한 종목 예시(숫자 넣어서)** →
  시간대별 가격/시그널/entry/exit를 표로 보여주고
* 그걸 **멀티 종목 행렬로 일반화한 코드 스케치**

까지 이어서 만들어 줄게.
