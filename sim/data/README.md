# 응답 길이 표본

`azure_output_histogram.json`은 Microsoft의 [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) 중 Conversation 파일에서 추출한 응답 토큰 수의 히스토그램이다.

- 출처: J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, E. Choukse, **DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency**, HPCA 2025. [논문](https://arxiv.org/abs/2408.00741).
- 원자료 라이선스: [CC BY 4.0](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE). 원자료와 이 변형 자료는 무보증으로 제공된다. Azure/Microsoft가 이 실험을 승인했다는 뜻은 아니다.
- 변형: 1,135,195,393바이트 원자료에서 균등 간격의 64 KiB 창 16개를 읽어 양끝의 불완전한 행을 제외했다. 각 창의 위치·SHA-256·관측 시각 범위는 JSON에 저장했다. 전체 자료나 요청별 균등 무작위 표본이 아니다.
- 포함된 양수 응답 길이: 25,202개. 평균 107.10, 중앙값 41, p95 465, p99 702, 최대 1,200 토큰. 표본의 관측 시각 범위는 2024-05-12~18이다.
- 추출 과정에서 원문 대화나 개인 식별자를 사용하지 않았다. 실험은 응답 길이를 독립적으로 재추출하며, 실제 대화 순서·입력 길이·턴 간 상관관계는 재현하지 않는다.
- `TIMESTAMP`는 요청 도착 시각이며 세션 식별자와 응답 완료 시각이 없으므로, 인접 행의 시각 차이를 사용자의 생각 시간으로 사용하면 안 된다. 실험의 생각 시간은 별도의 **합성 민감도 조건**이다.

재생성 (`sim/`에서 실행, 네트워크 필요):

```sh
python3 prepare_turn_workload.py
```

재실험은 저장된 히스토그램을 읽으므로 네트워크가 필요 없다. 실제로 측정한 짝 자료가 생기면 `output_tokens,think_s` 열을 가진 CSV를 `--turn-workload`에 전달할 수 있다. 이때 `think_s`는 해당 응답 완료부터 다음 사용자 요청까지의 간격이어야 하며, 입력 쌍은 실험 중 함께 표본화된다.
