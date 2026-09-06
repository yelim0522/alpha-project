# alpha-project

MEC(멀티액세스 엣지 컴퓨팅)를 메인 주제로 하는 논문 작업 저장소.

## 서브주제: 다중 사용자 이동성 환경 엣지 LLM 서빙의 조율형 선제 KV 마이그레이션

사용자가 이동해 서빙 엣지 서버(gNB)가 바뀔 때, 진행 중인 LLM 세션의 KV 캐시(대화 상태)를
새 서버에 어떻게 마련할지를 다룹니다.

- 반응형 계열(Full-Copy / Recomputation / ctHO)은 핸드오버 뒤에 복구를 시작하므로 복구
  시간이 그대로 서비스 중단(SIT)이 됩니다.
- Pallas(arXiv 2608.16477)는 이동 예측으로 핸드오버 전에 준비를 시작해(prefix 재계산 +
  suffix 스트리밍) 이 문제를 단일 사용자 관점에서 크게 해소했습니다.
- 그러나 Pallas의 스케줄러는 사용자별로 독립이며 경합을 관측치(EWMA)로만 인지합니다.
  플래툰처럼 다수 사용자가 같은 타겟을 향하면 모두 같은 순간에 트리거되어 첨두가
  사라지지 않고 시간축에서 앞으로 옮겨질 뿐입니다(herding). Pallas는 다중 사용자 자원
  배분을 명시적으로 범위 밖으로 두었습니다.

본 연구의 novelty는 "선제성"이 아니라 **타겟 단위로 여러 사용자의 프리페치 시점·suffix
처리 방식·prefill 순서·VRAM 승인·migrate/detour 선택을 공동 결정하는 조율**에 있습니다.
경합이 없으면 Pallas와 같은 결정으로 수렴하는, Pallas의 다중 사용자 일반화입니다.

- `paper/draft.md` — 논문 초안 v0.4(서론·관련연구·시스템 모델·herding 정식화·문제 정의·제안·보정/재현·밀도 스윕·ablation·최적해 격차·한계)
- `paper/references.md` — 참고문헌(주 베이스라인 Pallas, 보조 베이스라인 ctHO)
- `sim/` — 표준 라이브러리 기반 이산시간 시뮬레이터. Pallas 공개 수치(Table 1, Fig. 8(a))에 ±10%로 보정된 자원 모델, 6개 정책 + ablation 사다리, Pallas 재현/외삽 스크립트, 소규모 최적해 격차 스크립트

### 빠른 실행

```bash
cd sim && python3 run.py                       # 기본: 64명, 플래툰 8, 서버 6, Qwen3-32B
python3 run.py --sweep-users 32,64,128,192 --seeds 3   # 밀도 스윕(herding 재현)
python3 run.py --ablation --seeds 3            # 제안 메커니즘 사다리
python3 run.py --controlled --seeds 3          # Pallas 재조정 변형 통제실험
python3 reproduce_pallas.py                    # Pallas Table 1·Fig 8(a) 재현 + K>4 외삽
python3 optgap.py                              # K=3–5 전수 탐색 대비 격차
```
