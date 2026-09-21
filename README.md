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

EdgeFlow는 다중 KV 이전을, ImpactHO는 이동 사용자들의 부분 KV와 백홀 배분을 이미
다룹니다. 본 연구의 차별성은 **예측된 전환 시각 아래 완전한 상태를 준비할 때, 같은
목표 서버의 GPU·링크 경합을 고려해 여러 사용자의 준비 시작 시점을 공동 계획**하는
것입니다. 경합이 없으면 Pallas와 같은 결정으로 수렴합니다.

- [paper/draft.md](paper/draft.md) — 팀 검토용 초안 v0.12(현재 알고리즘 명세·보장 범위 정리, 기존 결과·그림 유지)
- [paper/coordination_algorithm.md](paper/coordination_algorithm.md) — 10줄 의사코드, 코드 대응, 메모리·최적성 한계와 기능별 비교 설계
- [paper/figures/README.md](paper/figures/README.md) — 논문용 그림 2개(PNG/PDF/SVG), 통계 정의와 재현 방법
- `paper/impactho_comparison.md` — ImpactHO와 본 연구의 문제·결정 변수·지표 대조
- `paper/turn_boundary_validation.md` — 턴 경계 한계 검증 결과, 응답 완료량·실제 대기·자원 점유 해석
- `paper/references.md` — 참고문헌(주 베이스라인 Pallas, 보조 베이스라인 ctHO)
- `sim/` — 표준 라이브러리 기반 이산시간 시뮬레이터. Pallas 공개 수치(Table 1, Fig. 8(a))에 ±10%로 보정된 자원 모델, 6개 정책 + ablation 사다리, Pallas 재현/외삽 스크립트, 소규모 최적해 격차 스크립트, 체제 지도(regime map) 스크립트

### 빠른 실행

```bash
cd sim && python3 run.py                       # 기본: 64명, 플래툰 8, 서버 6, Qwen3-32B
python3 run.py --sweep-users 32,64,128,192 --seeds 3   # 밀도 스윕(herding 재현)
python3 run.py --ablation --seeds 3            # 제안 메커니즘 사다리
python3 run.py --controlled --seeds 3          # Pallas 재조정 변형 통제실험
python3 reproduce_pallas.py                    # Pallas Table 1·Fig 8(a) 재현 + K>4 외삽
python3 optgap.py                              # K=3–5 전수 탐색 대비 격차
python3 regime_map.py --seeds 3                # 체제 지도: 선제형이 반응형에 지는 구간 확인
python3 alternatives.py --seeds 3 --csv alternatives.csv  # 턴 경계·헤징·KV 압축 비교
python3 turn_validation.py --seeds 3 --users 192 --seconds 600  # 정책별 대화 시계·길이 분포 검증
```

기존 결과로 그림만 다시 만들 때는 실험을 재실행할 필요가 없습니다. 별도 가상환경에
`sim/requirements-figures.txt`를 설치하고 저장소 루트에서
`python sim/plot_turn_validation.py`를 실행합니다([상세 절차](paper/figures/README.md)).
