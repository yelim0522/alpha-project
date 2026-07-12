# 엣지 LLM KV 캐시 핸드오버 시뮬레이터 (스켈레톤)

사용자 이동으로 서빙 엣지 서버가 바뀔 때, LLM 세션의 **KV 캐시**를
어떻게 이어줄지를 비교 평가하는 이산시간 시뮬레이터입니다.
표준 라이브러리만 사용하므로 `python run.py`로 바로 실행됩니다(설치 불필요).

## 실행

```bash
cd sim
python3 run.py
python3 run.py --users 120            # 고밀도(경합↑)
python3 run.py --users 80 --speed 40  # 고이동성
```

출력 예시:

```
policy                  HO  mean(s)  p99(s)  downtime  pingpong  prefetch
always-transfer       1518   11.936    22.4   18118.5        81         0
always-recompute      1518    1.706     2.0    2590.1        81         0
reactive-hybrid       1518    1.481   1.836    2248.2        81         0
predictive-prefetch   1518    1.415   1.836    2148.5        81     18925
```

## 모델 요약

- **상태 이전 비용**: KV 크기 = `c × 토큰수`. 핸드오버 지연은 재계산(prefill)과
  전송을 병렬 수행하므로 `max(p/v, c·(L−p)/B_eff)`. 최적 prefill 길이
  `p* = c·v·L / (B + c·v)` (자세한 유도는 `paper/draft.md` 3-2절).
- **백홀 경합(herding)**: 같은 시점에 핸드오버하는 사용자 수가 많을수록
  유효 대역폭 `B_eff = B / (1 + α·(n−1))`로 감소.
- **핑퐁**: 최근 `pingpong_window` 초 안에 떠났던 서버로 되돌아가는 핸드오버.

## 정책 (`policies.py`)

| 정책 | 설명 | 역할 |
|---|---|---|
| `always-transfer` | KV 전량 전송 (p=0) | 베이스라인 |
| `always-recompute` | KV 전량 재계산 (p=L) | 베이스라인 |
| `reactive-hybrid` | 핸드오버 시점에 최적 `p*` 선택 | 타겟 논문 근사 |
| `predictive-prefetch` | 이동 예측 선제 프리페치 + 적응적 재계산 + 핑퐁 게이팅 | **제안** |

## 지표 (`metrics.py`) — 초안 2 스타일 계층화

- **주**: 핸드오버 지연(mean/p99), 핑퐁 횟수(구조적 증상 직접 포착)
- **보조**: 프리페치 발동 수(개입-개선 인과)
- **부가**: 총 중단시간, 전송량, 핸드오버 수

## 파일 구조

- `environment.py` — 서버/사용자 엔티티, 비용 모델(핵심 수식)
- `mobility.py` — 랜덤 웨이포인트(실제 GPS 궤적으로 교체 가능)
- `policies.py` — 베이스라인 + 제안 정책
- `metrics.py` — 지표 수집/요약
- `run.py` — 시나리오 구성, 공통 트레이스 생성, 정책별 재생·비교

## 확장 지점(본실험 TODO)

1. **실제 이동성 트레이스**: `mobility.py`를 T-Drive/Rome/Porto taxi GPS 재생으로 교체.
2. **대화 길이 분포**: `run.py`의 토큰 증가/초기값을 LMSYS-Chat-1M 등 실측 분포로.
3. **프리페치의 백홀 비용 모델링**: 현재는 프리페치 전송량을 핸드오버 순간
   부하에서만 차감(선제 전송이 버스트를 분산한다는 효과를 근사). 프리페치 자체가
   소모하는 대역폭·경합을 명시적으로 반영하면 더 정밀해짐.
4. **다중 seed·신뢰구간**: `--seed`를 바꿔 반복하고 평균/신뢰구간 산출.
5. **시변 서버 용량**: `prefill_speed`/`backhaul_bw`를 시간에 따라 변동.

> ⚠️ 현재 수치는 **메커니즘 방향성 확인용(pilot)**이며, 파라미터에 따라 크기가
> 달라집니다. 절대 수치가 아닌 정책 간 상대 비교로 해석하세요.
