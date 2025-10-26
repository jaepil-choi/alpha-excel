## `add_data` (or `add_axis`) 필요성 토론 요약 🤔

### 1. 초기 의문 제기 (재필님의 관점)

* **핵심 질문:** `Field('market_cap')`처럼 `Expression` 내에서 데이터를 직접 참조하면 어차피 `EvaluateVisitor`가 데이터를 로드할 텐데, 왜 굳이 `rc.add_data('mcap', Field('market_cap'))`처럼 미리 "등록"하는 단계가 필요한가?
* **버킷 마스크 예시:** `size_expr = cs_quantile(Field('market_cap'), ...)`라고 정의했을 때, `mask_small = (size_expr == 'small')`처럼 `Expression` 자체에 비교 연산을 적용하여 불리언 마스크를 만들 수 있다면, `rc.add_axis('size', size_expr)`를 통해 `'size'`라는 중간 결과를 `rc.db`에 저장할 필요가 없어 보였다.

---

### 2. `add_data`/`add_axis`가 필요한 이유 (논의 결과)

`add_axis` (또는 `add_data_var`) 메서드가 단순히 마스크 생성을 넘어, **워크플로우**와 **효율성** 측면에서 중요한 역할을 한다는 결론에 도달했습니다.

1. **명시적 네이밍 및 재사용성 (Explicit Naming & Reusability) 🏷️:**
    * `rc.add_axis('size', size_expr)`는 `'small'`, `'big'`와 같은 **중간 계산 결과 (레이블)**를 `'size'`라는 **명확한 이름**으로 `rc.db` (`Dataset`)에 **저장**합니다.
    * 이렇게 저장된 `'size'` `DataArray`는 Fama-French 정렬뿐만 아니라, **종속 정렬(Dependent Sort)** 시 `group_by='size'` 인수로 참조되거나 다른 분석에서 **쉽게 재사용**될 수 있습니다.
    * `add_axis` 없이 `(size_expr == 'small')`만 사용하면, 이 중요한 중간 레이블 정보가 저장되지 않아 재사용이 어렵습니다.

2. **효율성 (Efficiency - Avoiding Recomputation) ⚡:**
    * `add_axis`를 사용하면 `size_expr` (`cs_quantile(...)`) 계산이 **단 한 번만** 수행되어 `rc.db['size']`에 캐시됩니다.
    * 이후 `mask_small = (rc.db['size'] == 'small')`이나 `mask_big = (rc.db['size'] == 'big')` 같은 연산은 이미 계산된 `DataArray`에 대한 **매우 빠른 비교 연산**입니다.
    * 반면, `mask_small = (size_expr == 'small')`, `mask_big = (size_expr == 'big')`를 각각 사용하면, `size_expr`(`cs_quantile`) 계산이 **여러 번 반복**되어 비효율적입니다.

3. **검증 및 디버깅 (Inspection & Debugging) ✅:**
    * `add_axis`를 통해 `rc.db['size']`를 직접 출력해보면, `cs_quantile` 연산이 의도한 대로 `'small'`, `'big'` 레이블을 올바르게 생성했는지 **중간 결과를 쉽게 확인**하고 검증할 수 있습니다.
    * `Expression`에 직접 비교 연산을 하면 이 중간 레이블링 결과를 확인하기 어렵습니다.

**결론:** `add_data`/`add_axis`는 워크플로우에서 **중간 결과를 명명하고, 재사용하며, 효율적으로 계산하고, 쉽게 검증**하기 위한 **매우 유용한 패턴**입니다. 특히 Fama-French처럼 여러 단계의 정렬과 조합이 필요한 경우 그 가치가 더욱 중요해집니다.
