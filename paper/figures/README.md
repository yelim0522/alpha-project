# 턴 경계 검증 그림

초안 v0.11 §5-9에 사용한 그림이다. 새 실험을 실행하거나 원자료를 수정하지 않고, 저장된 84회 결과를 다시 집계했다.

| 그림 | 확인할 내용 | 파일 |
|---|---|---|
| 1. 주요 조건 | 네 정책의 응답 대기 평균·p99, 추가 완료 시간 p99, 완료 응답 수 | [PNG](turn_boundary_main.png) · [PDF](turn_boundary_main.pdf) · [SVG](turn_boundary_main.svg) |
| 2. 민감도 | 7조건의 조율형/턴 경계 p99와 대응 시드별 완료량 비율 | [PNG](turn_boundary_robustness.png) · [PDF](turn_boundary_robustness.pdf) · [SVG](turn_boundary_robustness.svg) |

PNG는 문서 미리보기, PDF/SVG는 논문·발표용 벡터 그림이다. 그림 1과 2의 한국어 설명은 [초안](../draft.md)의 캡션을 따른다. 축과 범례는 글꼴 호환성을 위해 영어로 작성했다.

## 자료와 통계

- 입력: `sim/turn_validation_v09_seeds.csv`, `_summary.csv`, `_manifest.json`. 사용자 192명, 600초, 시드 0·1·2, 4정책 × 7조건이다.
- 막대/큰 점은 **시드별 통계의 평균**, 작은 점은 개별 시드이다. 그림 2의 오차 막대는 **표본 표준편차**이며 신뢰구간이 아니다. p99도 시드별 p99의 평균이며 응답을 합친 p99나 최악 사용자 지표가 아니다.
- 그림 2의 비율은 같은 시드의 `턴 경계 완료량 / 조율형 완료량`을 구한 뒤 평균한다. 본문의 11.02%는 `1 − 두 정책의 시드 평균 완료량 비율`이다. 두 계산을 구분한다. 대응 값은 [CSV](turn_boundary_paired.csv)에 저장했다.
- 지연 통계는 완전히 관측한 완료 응답만 포함한다. 작업 완료 수와 검열 표본이 정책별로 달라 동일 작업 집합의 지연 비교가 아니다. [상세 보고서](../turn_boundary_validation.md)의 미완료·관측 대기를 함께 본다.
- 추가 완료 시간은 `요청부터 마지막 토큰까지 시간 − 출력 토큰 수 / 기본 생성 속도`다. 첫 토큰 지연과 다르다.
- 실측은 Azure **응답 길이 주변 분포**에 한정된다. 생각 시간은 합성이며 실제 대화 재생이 아니다. 생성 예산 240 token/s는 미보정이고, 0.25초 조건은 이동·예측 격자도 바뀐다.

## 재현

저장소 루트에서 실행한다. 시뮬레이터의 표준 라이브러리 환경과 분리한 임시 가상환경을 쓴다.

```sh
figure_env=$(mktemp -d /private/tmp/alpha-figures.XXXXXX)
python3 -m venv "$figure_env/venv"
"$figure_env/venv/bin/python" -m pip install -r sim/requirements-figures.txt
MPLCONFIGDIR="$figure_env/mpl" XDG_CACHE_HOME="$figure_env/cache" \
  "$figure_env/venv/bin/python" sim/plot_turn_validation.py
python3 -m unittest discover -s sim -v
```

출력은 이 디렉터리의 생성 파일들을 갱신한다. 다른 곳에 저장하려면 `--output-dir /원하는/경로`를 사용한다. 시뮬레이션 재실행은 필요 없다. Python 3.9 이상과 matplotlib 3.9.4를 사용했다.

그리기 전에 전체 조건·시드 조합, 중복/누락, 유한한 비음수 지표, 대기 회계, **모든 요약 열의 평균·표준편차**, 실험 당시 코드·작업 자료 해시를 검사한다. 코드 해시가 달라지면 과거 결과를 현재 코드의 결과처럼 그리지 않도록 중단한다. 이 경우 과거 실험 버전에서 재현하거나 변경된 코드로 별도 실험을 수행해야 한다. 검증 기록·입력 해시·그림 코드 해시는 [그림 생성 기록](turn_boundary_plot_manifest.json)에 있다.

## 제출 전 남은 점검

이 그림은 시뮬레이션 결과 정리이며 실제 시스템 우월성의 증거가 아니다. 생각 시간·GPU 실행·전체 VRAM의 검증, 이전 경로별 자원 모델 차이, 제출처 형식과 참고문헌 원문 대조는 여전히 남아 있다. 기존 §5-3–5-8 수치를 이번에 모두 재실행한 것은 아니다.
