# 엣지 LLM KV 캐시 핸드오버 시뮬레이터

사용자 이동으로 서빙 엣지 서버(gNB)가 바뀔 때 LLM 세션의 KV 캐시를 어떻게 이어줄지를,
**공유 자원(타겟 GPU prefill·gNB 간 백홀 링크·VRAM)** 위에서 비교 평가하는 이산시간
시뮬레이터입니다. 표준 라이브러리만 사용하므로 `python3 run.py`로 바로 실행됩니다.

## 실행

```bash
cd sim
python3 run.py                                  # 64명, 플래툰 8, 서버 6, 300 Mbps
python3 run.py --mobility rwp                   # 독립 이동(랜덤 웨이포인트)
python3 run.py --users 128 --group 16           # 고밀도 플래툰(herding 강화)
python3 run.py --sweep-users 16,32,64,128       # 밀도 스윕
python3 run.py --controlled --seeds 3           # Pallas/제안 파라미터 변형 통제실험
python3 run.py --pred-heading-noise 0.2 --pred-speed-noise 0.2   # 예측 잡음 스트레스
```

출력 열: `HO` 핸드오버 수, `SITavg/p99/max` 서비스 중단 시간(s), `ITLms` 평균 토큰 간
지연(Detour 비용), `pkStrm` 링크별 동시 스트림 첨두, `pkMB/s` 링크 요구량 첨두,
`CoV` 시간축 링크 요구량 변동계수, `early` 조기 준비 노출 평균(s), `wasteMB` 취소로
낭비된 준비량, `xferMB` 총 전송량, `detour`, `pingpong`.

## 모델 요약

기본 파라미터는 Pallas 논문의 공개 수치로 보정했습니다(Qwen3-32B 256 KiB/토큰,
prefill ≈ 2,000 tok/s, decode 15 tok/s, 300 Mbps, T0 = 150 ms, α = 0.8, T_max = 5 s).

- **단일 사용자 비용**: 반응형은 `max(p/v, c·(L−p)/B)`, 최적 분할 `p* = c·v·L/(B+c·v)`.
  선제형은 Pallas 식 (4)–(8): 안정 prefix는 타겟에서 prefill, window 동안 자라는 suffix는
  스트리밍, `T_SIT = T0 + max{0, T_hist − T_w, T_res}`, `T_early = max{0, T_w − T_hist}`.
- **공유 자원**(`environment.advance_preparations`): 타겟 GPU는 `v_m`을 정책이 정한 순서
  (FCFS 또는 EDF)로 준비 중인 prefix들에 분배, 링크는 백로그가 있는 스트림에 `B_m`을
  공정 분배. 핸드오버 시 잔여 복구는 그 순간 같은 타겟에서 진행 중인 준비·복구 수로
  나눈 유효율을 받습니다. **선제 프리페치가 소모하는 대역폭·GPU가 명시적으로
  반영됩니다.**
- **예측**(`prediction.py`): 등속·등방향(CVH) 외삽. 잡음 옵션으로 예측 품질 스트레스.
- **이동**(`mobility.py`): `GroupFlow`(플래툰 직선 흐름; 영역 재진입은 새 세션으로
  처리), `RandomWaypoint`. 실제 GPS 궤적(nuScenes, T-Drive 등) 재생으로 교체 가능.
- **Detour**: 상태를 소스에 두고 토큰을 포워딩. 홉당 고정 ITL 페널티.

## 정책 (`policies.py`)

| 정책 | 설명 | 역할 |
|---|---|---|
| `full-copy` | KV 전량 전송 (p=0) | 베이스라인 |
| `recompute` | KV 전량 재계산 (p=L) | 베이스라인 |
| `ctho-approx` | 핸드오버 시점 최적 분할, 다중 사용자 공정 분배 | 반응형 joint 근사 |
| `detour` | 소스 유지 + 포워딩 | 연속성 극단 |
| `pallas-approx` | 사용자별 window 그리드 탐색(관측 EWMA 유효율), prefix FCFS + suffix 스트리밍, 타겟 변경 시 취소 | **주 베이스라인** |
| `coordinated` | 타겟 단위 공동 계획(EDF 순 staggering, 동시성 상한 `k_max`), 계획 동시성 기반 비용, suffix 모드 선택(stream/defer), EDF prefill, VRAM 승인, 예측 불일치·핑퐁 시 detour + settle | **제안** |

`--controlled`는 Pallas 변형(`T_max`=15, `α`=0.5, EWMA 0.95)과 제안 변형(동시성 상한
제거, 스트림 임계 완화)을 추가하여, 이득이 파라미터 조정이 아닌 결정 구조에서 오는지
분리합니다.

## 지표 (`metrics.py`)

- **주**: SIT 평균/p99/최대, ITL 평균
- **구조(다중 사용자)**: 동시 스트림 첨두, 링크 요구량 첨두, 시간축 CoV, T_early, 낭비량
- **부가**: 총 전송량, detour/핑퐁/핸드오버 수

## 파일 구조

- `environment.py` — 서버/사용자/준비(Prep) 엔티티, 단일 사용자 비용식, 공유 자원 진행
- `prediction.py` — CVH 이동 예측
- `mobility.py` — GroupFlow / RandomWaypoint
- `policies.py` — 베이스라인 + Pallas 근사 + 제안
- `metrics.py` — 지표 수집/요약
- `run.py` — 시나리오 구성, 공통 트레이스 생성, 정책별 재생·비교

## 루프 순서(주의)

각 스텝은 (1) 직전 스텝의 예측과 자원 상태로 이번 스텝에 감지된 핸드오버를 먼저 처리하고,
(2) 새 예측으로 정책의 제어 주기를 돌린 뒤, (3) 준비를 한 스텝 진행합니다. 순서를 바꾸면
핸드오버 스텝의 예측이 이미 다음 셀을 가리켜 준비가 오인 취소되므로 선제 정책이 준비를
전혀 활용하지 못하게 됩니다.

## 확장 지점(본실험 TODO)

1. **Pallas 재현 검증**: K=1–4 동시 마이그레이션(Pallas Fig. 8a)과 Table 1·2 수치를
   시뮬레이터가 재현하는지 먼저 확인해 보정 신뢰를 확보.
2. **실제 이동성 트레이스**: nuScenes(Pallas와 동일), T-Drive/Rome/Porto taxi.
3. **대화 길이 분포**: ShareGPT/LMSYS-Chat-1M 실측 분포.
4. **자원 모델 정밀화**: chunked prefill과 상주 디코딩 세션(HI 경합), 링크 프로토콜 오버헤드.
5. **예측 신뢰도 활용**: 다중 후보 확률 가중 준비(헤징).
6. **공정성**: worst-user SIT, Jain 지수.
7. **다중 seed·신뢰구간**: `--seeds N`.

> ⚠️ 현재 수치는 **메커니즘 방향성 확인용(pilot)**입니다. 절대 수치가 아닌 정책 간 상대
> 비교로 해석하세요.
