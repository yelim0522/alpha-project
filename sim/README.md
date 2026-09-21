# 엣지 LLM KV 캐시 핸드오버 시뮬레이터

사용자 이동으로 서빙 엣지 서버(gNB)가 바뀔 때 LLM 세션의 KV 캐시를 어떻게 이어줄지를,
**공유 자원(타겟 GPU prefill·gNB 간 백홀 링크·VRAM)** 위에서 비교 평가하는 이산시간
시뮬레이터입니다. 표준 라이브러리만 사용하므로 `python3 run.py`로 바로 실행됩니다.

논문용 그림은 선택 의존성 `requirements-figures.txt`를 설치한 별도 환경에서
`python3 plot_turn_validation.py`로 생성합니다. 저장된 84회 결과와 원본 구현 해시를
검증한 뒤 PNG/PDF/SVG를 만듭니다. [그림·재현 안내](../paper/figures/README.md)를 참고하세요.

## 실행

```bash
cd sim
python3 run.py                                  # 64명, 플래툰 8, 서버 6, 300 Mbps, Qwen3-32B
python3 run.py --sweep-users 32,64,128,192 --seeds 3          # 밀도 스윕
python3 run.py --ablation --seeds 3             # 제안 메커니즘 사다리(ablation)
python3 run.py --controlled --seeds 3           # Pallas/제안 파라미터 변형 통제실험
python3 run.py --pred-heading-noise 0.2 --pred-speed-noise 0.15   # 예측 잡음 스트레스
python3 run.py --mobility rwp                   # 독립 이동(herding 없음)
python3 run.py --model llama8b                  # 모델 프리셋(qwen32b/qwen14b/llama8b)
python3 run.py --users 12 --group 1             # 저경합: 제안 = Pallas 수렴 확인
python3 reproduce_pallas.py                     # Pallas Table 1·Fig 8(a) 재현 + K>4 외삽
python3 optgap.py                               # 소규모 인스턴스 최적해 대비 격차(brute force)
python3 regime_map.py --seeds 3 --csv out.csv   # 체제 지도: 문맥×사용자, 잡음×사용자, 백홀×문맥 (약 30분)
python3 alternatives.py --seeds 3 --csv alternatives.csv  # 턴 경계·다중 후보 헤징·이상적 KV 압축
python3 alternatives.py --maps T --turn-only --seeds 3 --csv alternatives_turn.csv  # 턴 경계만 재검증
python3 turn_validation.py --seeds 3 --users 192 --seconds 600  # 정책별 대화 시계·실측 응답 길이
```

출력 열: `HO` 핸드오버 수, `SITavg/p99/max` 서비스 중단 시간(s), `prep%` 준비를 사용한
마이그레이션 비율, `SITprep/prp99` 준비 사용 마이그레이션의 SIT 평균/p99, `SITnoprp`
반응형으로 떨어진 경우의 SIT, `ITLms` 평균 토큰 간 지연(Detour 비용), `pkStrm` 링크별
동시 스트림 첨두, `CoV` 시간축 링크 요구량 변동계수, `Jain` 사용자별 평균 SIT의 Jain
공정성 지수, `early` 조기 준비 노출 평균(s), `wasteMB` 취소로 낭비된 준비량, `detour`.

## 모델 요약

### 보정(`reproduce_pallas.py`)
파라미터는 Pallas 논문의 공개 수치에 맞춥니다. Table 1(Qwen3-32B, 300 Mbps, 1K/2K/4K
토큰의 Full-Copy/Recomputation/ctHO)로 `v1`·`T0(반응형)`을, Fig. 8(a)(Qwen3-14B, 2K
토큰, 1 Gbps, K=1–4 동시 UE)로 `T0(선제형)`·활성화 직렬화 비용을 맞추면 모든 셀이
대부분 ±10%(K=3 평균 +15%, 최대 +12% 예외) 안에 들어옵니다. 같은 스크립트가 K=6–32로 외삽하여, 원 논문의
측정 범위 밖에서 비조율 Pallas와 조율 정책이 어떻게 갈라지는지
보여줍니다.

### 자원 모델(`environment.py`)
- **GPU prefill 2단계**: 요청 하나는 최대 `v1` tok/s(모델별 프리셋), 여러 요청은 배칭으로
  합계 `prefill_parallel × v1`까지. 즉 동시 prefill이 `P`(기본 4)개까지는 서로 느려지지
  않고 그 이상이면 균등 분배됩니다. Pallas Fig. 8(a)에서 K≤4일 때 SIT가 거의 평평한 것이
  이 형태를 요구합니다.
- **활성화 비용 분리**: 반응형 복구의 `T0`(75 ms)와 준비된 요청의 활성화 `T0^p`(236 ms;
  최종 동기화·조립)를 구분하고, 같은 주기에 여러 준비가 활성화되면 요청당 `40 ms`씩
  직렬화됩니다(Fig. 8(a) K=4 worst-user 359 ms에서 맞춤).
- **공유 자원**(`advance_preparations`): 타겟 GPU는 정책이 정한 순서(FCFS 또는 EDF)로
  준비 중인 prefix에 배분, 링크는 백로그가 있는 스트림에 `B_m`을 공정 분배. 핸드오버
  시점의 반응형 복구는 그 순간 같은 타겟에서 복구 중인 사용자 수(EDF 정책이면 진행 중인
  준비는 선점, FCFS면 함께 대기)로 나눈 유효율을 받습니다. 준비된 prefix의 활성화는
  prefill이 아니므로 GPU 지분을 희석하지 않습니다.
- **단일 사용자 비용**: 반응형 `max(p/v, c·(L−p)/B)`, 최적 분할 `p* = c·v·L/(B+c·v)`.
  선제형은 Pallas 식 (4)–(8).
- **예측**(`prediction.py`): 등속·등방향(CVH) 외삽, 시간 격자 = 제어 주기. 잡음 옵션.
- **이동**(`mobility.py`): `GroupFlow`(플래툰 직선 흐름; 영역 재진입은 새 세션),
  `RandomWaypoint`. 실제 GPS 궤적 재생으로 교체 가능.
- **Detour**: 상태를 소스에 두고 토큰을 포워딩. 홉당 고정 ITL 페널티.

## 정책(`policies.py`)

| 정책 | 설명 | 역할 |
|---|---|---|
| `full-copy` | KV 전량 전송 (p=0) | 베이스라인 |
| `recompute` | KV 전량 재계산 (p=L) | 베이스라인 |
| `ctho-approx` | 핸드오버 시점 최적 분할, 다중 사용자 공정 분배 | 반응형 joint 근사 |
| `detour` | 소스 유지 + 포워딩 | 연속성 극단 |
| `pallas-approx` | 사용자별 window 그리드 탐색(관측 EWMA 유효율), prefix FCFS + suffix 스트리밍, 타겟 변경 시 취소 | **주 베이스라인** |
| `coordinated` | 타겟 단위 공동 계획: 계획된 GPU 점유 프로파일로 window별 완료 시각을 유체 모델(EDF 인지)로 계산하고, 다른 준비에 주는 지연(외부효과, 가중 β)을 비용에 더해 트리거를 시간축으로 분산. 계획은 예측이 바뀌기 전까지 유지. suffix 모드 선택(stream/defer), EDF prefill, VRAM 승인, 예측 불일치·핑퐁 시 detour + settle | **제안** |
| `turn-boundary` | 현재 턴은 소스에서 포워딩하고 생성 종료 후 think interval에 ctHO 복구. think time을 넘는 부분만 SIT | **대안 비교군** |
| `pallas-hedge2` | noisy-CVH ensemble의 상위 2개 타겟에 prefix/suffix를 모두 준비. 복제본도 실제 GPU·백홀·VRAM을 소모하고 loser는 낭비로 계산 | **대안 비교군** |

두 선제 정책은 같은 window 격자(0.2 s), 같은 목적함수(α=0.8), 같은 `T_max`(5 s), 같은
트리거 규칙을 쓰며, 후보가 하나이고 활성 준비가 없으면 `coordinated`는 `pallas-approx`와
**동일한 결정**을 냅니다(`--users 12 --group 1`에서 수치가 일치).

- `--ablation`: `abl-planned-k`(계획 점유 기반 완료 시각만) → `+social`(외부효과 항) →
  `+suffix`(모드 선택) → `+edf`(EDF 서비스; detour 없는 완성형) → `+detour`(= `coordinated`).
- `--controlled`: Pallas 변형(`T_max`=15, `α`=0.5, EWMA 0.95, EDF 서비스만 추가)과 제안
  변형(외부효과 항 제거, β=0.5/2/4, `T_max`=15, 스트림 임계 완화).

## 지표(`metrics.py`)

- **주**: SIT 평균/p99/최대, ITL 평균, 준비 사용/미사용 분리 SIT
- **구조(다중 사용자)**: 동시 스트림 첨두, 링크 요구량 첨두, 시간축 CoV, T_early, 낭비량
- **공정성**: 사용자별 평균 SIT의 Jain 지수
- **부가**: 총 전송량, detour/핑퐁/핸드오버 수

## 파일 구조

- `environment.py` — 서버/사용자/준비(Prep) 엔티티, 모델 프리셋, 단일 사용자 비용식, 공유 자원 진행
- `prediction.py` — CVH 이동 예측
- `mobility.py` — GroupFlow / RandomWaypoint
- `policies.py` — 베이스라인 + Pallas 근사 + 제안(+ ablation 사다리)
- `metrics.py` — 지표 수집/요약
- `run.py` — 시나리오 구성, 공통 트레이스 생성, 정책별 재생·비교
- `minisim.py` — 단일 타겟 미니 시뮬레이터(재현·최적해 격차 공용)
- `reproduce_pallas.py` — Pallas Table 1 / Fig. 8(a) 재현, K>4 외삽
- `optgap.py` — K=3–5 인스턴스에서 트리거 스케줄 전수 탐색 대비 격차
- `regime_map.py` — 문맥 길이·사용자 수·예측 잡음·백홀 대역폭 격자에서 ctHO/Pallas/제안의 승자 지도.
  셀마다 SIT 평균·p99, 비율 격자(Pallas/ctHO, 제안/Pallas), Detour의 ITL 페널티를 출력(`--csv`)
- `alternatives.py` — (T/T2) 턴 경계 및 think-time 민감도, (H) 예측 잡음별 top-2 헤징,
  (K) 이상적 KV 압축률(1, 1/2, 1/4, 1/8) 비교. 압축은 코덱 오버헤드를 빼 압축에
  유리한 upper bound이며, 턴 모델 기본값은 output 128 tokens / think 4 s이다.

## 루프 순서(주의)

각 스텝은 (1) 직전 스텝의 예측과 자원 상태로 이번 스텝에 감지된 핸드오버를 먼저 처리하고,
(2) 새 예측으로 정책의 제어 주기를 돌린 뒤, (3) 준비를 한 스텝 진행합니다. 트리거 규칙은
"이상적 트리거 시점이 다음 제어 주기 전에 오면 지금 트리거"로 이산화했습니다(격자와 예측
잔여 시간이 어긋나 트리거가 영구히 누락되는 문제를 막음).

## 정책별 대화 시계 검증

대화 시계 검증은 `--conversation-clock closed-loop`로 활성화한다. 기본 `trace` 모드는 기존 체제 지도와 비교를 위해 유지한다. 새 모드는 모든 정책에서 실제 복구 대기와 전달 지연만큼 다음 턴을 늦추며, 완료 응답 수와 검열된 대기도 보고한다. `response_wait_*`는 완료 응답별 실제 중단 시간이고 기존 이전 사건별 SIT와 분모가 다르다. `response_excess_*`는 중단과 전달 지연을 포함한 응답 완료까지의 추가 시간이다.

```sh
python3 run.py --alternatives --conversation-clock closed-loop --turn-distribution lognormal --think-distribution mixture
python3 run.py --alternatives --conversation-clock closed-loop --turn-workload data/azure_output_histogram.json
python3 turn_validation.py --seeds 1 --users 16 --seconds 60 --scenarios fixed,azure-mixture --output /tmp/turn_smoke
python3 -m unittest discover
```

`turn_validation.py`는 시드별 CSV·평균/표준편차 CSV·재현용 설정/해시 JSON을 남긴다. 저장된 Azure 히스토그램을 쓰므로 실험은 오프라인 실행 가능하다. 자료 추출은 `prepare_turn_workload.py`, 출처·표본 한계는 `data/README.md`, 결과는 `../paper/turn_boundary_validation.md`를 참조한다. 생각 시간은 합성 가정이고 실제 대화 순서를 재생하지 않는다. `--decode-capacity`는 서버당 생성 처리량 민감도이며 GPU 실측값이 아니다. VRAM은 점유를 기록하되 전체 상주 메모리의 승인·퇴거는 모델링하지 않는다.

## 확장 지점(본실험 TODO)

1. **실제 이동성 트레이스**: nuScenes(Pallas와 동일), T-Drive/Rome/Porto taxi.
2. **대화 길이·생각 시간**: Azure 응답 길이 표본과 합성 생각 시간 검증은 추가됨. 실제 세션 순서·입력 토큰·턴 간 간격의 짝 자료가 남음.
3. **자원 모델 정밀화**: chunked prefill과 상주 디코딩 세션(HI 경합), 링크 프로토콜 오버헤드.
4. **VRAM을 계획에 반영**: 현재는 트리거 시점 승인만 하므로 예산이 작으면 준비가 보류됨.
5. **예측 신뢰도 활용**: 다중 후보 확률 가중 준비(헤징).
