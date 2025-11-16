좋은 포인트였고, 방향이 훨씬 “엔진 레벨”에 가깝다.
아래는 **“price·shares 기반 현실적인 vectorized 백테스트 엔진”**에 맞게 다시 쓴 리포트다.
전제: **시그널 가공(랭킹, 중립화 등)은 이미 리서처가 끝냈고**, 엔진은 “얼마나, 몇 주를 사고팔지”만 처리한다.

---

# 1. 문제의식: weight 기반 vectorized 백테스트의 한계

많은 교과서식 백테스트는 이렇게 가정한다:

* 시그널 행렬 (W_{t,i}): 각 시점·종목별 **포트폴리오 비중(weight)**
* 수익률 행렬 (R_{t,i})
* PnL:
  [
  \text{PnL}*t = \sum_i W*{t-1,i} \cdot R_{t,i}
  ]

하지만 이 방식은 엔진 관점에서 여러 가지 문제가 있다.

* 실제로는 **정수 개의 주식**을 거래해야 하지만,
  weight 기반 모델은 **“0.123456 주 보유”** 같은 비현실적인 상태를 허용한다.
* **현금(cash) 잔고, 증거금, 계좌 단위 자산**이 명시적으로 존재하지 않는다.
* 거래는 모두 “연속량(continuous)”으로 처리되어,
  lot size, 최소 주문 단위 등 현실적인 제약을 반영하기 어렵다.
* 결과적으로, 이 레벨의 로직은
  **signal processing / portfolio construction 단계에 더 가깝고**
  “백테스팅 엔진”보다는 “이론적 factor 시뮬레이션”에 가깝다.

따라서 엔진 설계의 관점에서는,
**price·shares·cash 기반의 discrete 백테스트**가 더 적절하다.

---

# 2. 목표: price & shares 기반 vectorized 엔진

이 리포트의 목표는 다음과 같다.

> **주어진 시그널 행렬을 바탕으로,
> 각 시점에 “몇 주를 어떤 가격에 사고팔았는지”를 vectorized하게 계산하고,
> 그로부터 포지션·캐시·PnL 타임시리즈를 만드는 엔진 설계.**

전제:

* 시그널 행렬 (S_{t,i})는 이미 가공된 “트레이딩 신호”다. 예:

  * (S_{t,i} \in {-1, 0, +1}):

    * +1: 새로/추가 매수
    * −1: 새로/추가 매도(혹은 숏)
    * 0: 아무것도 안 함
  * 또는 “사야 할 단위 수(contracts/units)” 형태 등
* 엔진은 **S를 그대로 받아, price·shares 세계에서 처리**한다.

---

# 3. 데이터 구조와 기본 수식

엔진에서 다루는 기본 행렬·벡터는 다음과 같다.

* (P_{t,i}): 가격 행렬 (일봉·분봉 무엇이든 가능)
* (S_{t,i}): 시그널 행렬 (매수/매도/홀드 등)
* (Q_{t,i}): 보유 수량(share/contract) 행렬
* (C_t): 캐시(현금) 타임시리즈
* (\text{PNL}_t): 계좌 기준 일일 PnL

핵심 관계는:

1. **트레이드 수량**
   [
   \Delta Q_{t,i} = Q_{t,i} - Q_{t-1,i}
   ]

2. **트레이드 금액(체결대금)**
   [
   Notional_{t,i} = \Delta Q_{t,i} \cdot Price^{\text{exec}}_{t,i}
   ]

3. **캐시 변화**
   [
   C_t = C_{t-1} - \sum_i Notional_{t,i} - \text{Fees}_t
   ]

4. **포지션 평가손익(MtM PnL)**
   [
   \text{PNL}*t^{\text{pos}} = \sum_i Q*{t-1,i} \cdot (P_{t,i} - P_{t-1,i})
   ]

5. **총 PnL**
   [
   \text{PNL}_t = \text{PNL}_t^{\text{pos}} - \text{Fees}_t
   ]

이 수식들은 대부분 **행렬 연산과 축 방향 합(sum over axis)**으로 표현 가능하다.

---

# 4. 시그널 → 정수 수량(share)로 변환

핵심은:

> **시그널 (S_{t,i})를 그대로 weight로 쓰지 않고,
> “얼마나(몇 주) 매수/매도할 것인가”를 price와 계좌 규모를 이용해 계산하는 것.**

## 4.1 단위 리스크/단위 금액 기반 sizing

예시: 각 시그널 +1에 대해 “종목 하나당 K 원어치씩 산다”라고 정한다.

1. 계좌 자산(Equity)를 (E_{t-1})라고 할 때,
   per-signal 단위 금액을:
   [
   A_t = \alpha \cdot E_{t-1}
   ]
   (예: 전체 자산의 1%씩)

2. 매수해야 할 “목표 주식 수”는:
   [
   Q^{target}*{t,i} =
   \left\lfloor
   \frac{A_t \cdot \max(S*{t,i}, 0)}{P_{t,i} \cdot LotSize_i}
   \right\rfloor \cdot LotSize_i
   ]

   * `max(S,0)`는 buy 시그널만 반영 (sell은 별도 처리)
   * `LotSize_i`는 어떤 시장에서는 1주, 어떤 시장은 10주 단위 등
   * floor를 쓰기 때문에 **정수 개(lot 단위)** 의 주식만 산다.

3. 매도·숏 포지션도 같은 방식으로,

   * −1 시그널일 때는 현재 보유 수량과 margin 제약을 고려해
     정수 개로 계산한다.

이 모든 계산은 **P, S, E, LotSize**가 같은 shape를 가진 행렬이라면
브로드캐스팅으로 한 번에 처리 가능하다.

## 4.2 기존 포지션과의 차이 → 실제 주문 수량

위에서 계산한 것은 “목표 수량(target holdings)”이다.
실제 주문 수량은:

[
\Delta Q_{t,i} = Q^{target}*{t,i} - Q*{t-1,i}
]

이를 통해:

* 신규 매수, 부분 청산, 전량 청산
* 숏 진입, 숏 축소/청산

이 모두가 **정수 share 기준**으로 처리된다.

---

# 5. 거래 비용 및 유동성 모델 (share 단위)

정수 share 기반에서는 비용·용량 계산도 자연스럽다.

## 5.1 거래 비용

* 거래대금:
  [
  Notional_{t,i} = |\Delta Q_{t,i}| \cdot Price^{\text{exec}}_{t,i}
  ]
* 비용:
  [
  Fees_t = \sum_i \left( c_i^{\text{fixed}} + c_i^{\text{bps}} \cdot Notional_{t,i} \right)
  ]

여기서:

* (c_i^{\text{bps}})는 종목별 수수료율(수수료 + 평균 스프레드 + 슬리피지 근사)
* 이 역시 `abs`, `*`, `sum`으로 벡터화 가능하다.

## 5.2 유동성 / capacity 제약

* ADV(일평균 거래대금) 행렬 (ADV_{t,i})를 가지고 있다면,
  아래와 같은 제약을 둘 수 있다:

[
|Notional_{t,i}| \le k \cdot ADV_{t,i}
]

이 제약을 만족하도록 `ΔQ_{t,i}`를 비율로 줄이거나,
아예 0으로 만드는 것도 벡터식 `clip`·`where`로 처리 가능하다.

---

# 6. 타이밍 및 체결 가격 가정

price 기반 엔진에서는 **“어떤 가격으로 체결되었다고 가정할 것인가”**가 명확해진다.

## 6.1 시그널 lag

기본적으로:

* (S_{t-1,i}): t-1까지의 정보를 기반으로 만든 시그널
* t 시점의 바(분봉/일봉) 시작 가격에서 체결

따라서:

[
\Delta Q_{t,i} = f(S_{t-1,i}, P_{t}^{open}, E_{t-1}, \dots)
]

이렇게 하면 look-ahead bias를 피할 수 있다.

## 6.2 체결 가격

단순화 예:

* 진입·리밸런싱: t 시가 (`P^{open}_t`)
* 포지션 평가: t 종가 (`P^{close}_t`)
  → 일일 수익률은 `close / prev_close - 1`,
  체결대금은 `ΔQ * open`으로 계산.

intraday(분봉)에서도 동일하게:

* 이전 바까지의 정보를 보고, 다음 바의 `open`에 체결했다고 가정
* 해당 바의 `close`까지 보유

---

# 7. 리스크 및 레버리지 관리 (shares + price 기반)

weight가 아닌 **shares·price·equity** 기반으로도
리스크 관리를 벡터화할 수 있다.

## 7.1 계좌 자산과 레버리지

계좌 자산(Equity):

[
E_t = C_t + \sum_i Q_{t,i} \cdot P_{t,i}
]

총 익스포저(notional):

[
Exposure_t = \sum_i |Q_{t,i} \cdot P_{t,i}|
]

레버리지:

[
Leverage_t = \frac{Exposure_t}{E_t}
]

이 값이 어떤 한도를 넘으면:

* 다음 시점 주문에서 target 수량을 줄이거나,
* 전체 포지션을 proportionally 스케일 다운하는 등
  벡터 연산으로 제어할 수 있다.

## 7.2 볼 타게팅

전략 수익률:

[
r_t = \frac{E_t - E_{t-1}}{E_{t-1}}
]

롤링 변동성 추정:

[
\hat{\sigma}*t = \text{Std}(r*{t-L+1 .. t})
]

타깃 변동성 (\sigma^*)에 맞춰
다음 시점 주문 때 단위금액 (A_t)를 조절:

[
A_t = A_0 \cdot \frac{\sigma^*}{\hat{\sigma}_t}
]

→ 신호는 동일하더라도 “몇 주 살지”를 조절하여
전체 전략의 변동성을 관리할 수 있다.

---

# 8. 정리

* **weight 기반 vectorized 백테스트**는
  이론 연구에는 편하지만,

  * 정수 개의 종목 매매
  * 캐시/레버리지/용량
    를 명확히 반영하지 못한다.

* 보다 엔진다운 접근은:

  1. **시그널 행렬 (S_{t,i})** 는 이미 리서처가 만든 “트레이딩 신호”라고 보고,
  2. 이를 price·equity·lot size를 이용해
     **정수 share 수량 (Q_{t,i})** 로 변환하며,
  3. 캐시, 포지션, PnL을 모두 **price·shares·fees** 기반으로 계산하는 것이다.

* 이 과정은 여전히:

  * 종목 축에 대해서는 완전히 벡터화되어 있고,
  * 시간 축에 대해서도 `cumsum`, `rolling` 등을 이용해
    대부분 행렬 연산으로 처리할 수 있다.

결과적으로,
**“weight 없는, price·shares 기반 vectorized 엔진”**은
이론적인 factor 시뮬레이터와
실제 주문/체결을 다루는 이벤트 드리븐 엔진 사이를 잇는
현실적인 중간 단계 역할을 할 수 있다.

필요하다면,
지금 리포트에 맞춰서 **실제 수식 → NumPy/Pandas 코드 스켈레톤**도 만들어 줄게.
