# 참고문헌 (24~26년 중심)

> 본문 인용 키와 매핑. 실제 제출 전 각 항목의 서지정보(권/호/페이지/DOI)를 원문에서 재확인할 것.
> arXiv 프리프린트는 저널/학회 게재 확정 시 정식 서지로 교체 권장.

## 직접 관련 (타겟 및 핸드오버)

- [target] S. Lee, J. Park, C. Zheng, and H. Park, "Low-Latency Edge LLM Handover via Joint KV Cache Transfer and Token Prefill," *arXiv preprint arXiv:2603.28018*, 2026. — 타겟 논문
- [ilcp] "Inductive Latent Context Persistence: Closing the Post-Handover Cold Start in 6G Radio Access Networks," *arXiv preprint arXiv:2605.00593*, 2026.
- [dapo] "DAPO: Mobility-Aware Joint Optimization of Model Partitioning and Task Offloading for Edge LLM Inference," *Electronics*, vol. 14, no. 19, Art. no. 3929, 2025. (모델 분할 축 — 대비용)

## 엣지 LLM 서빙 / 모델 캐싱 (배경)

- [survey1] G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, "Mobile Edge Intelligence for Large Language Models: A Contemporary Survey," *IEEE Communications Surveys & Tutorials*, vol. 27, no. 6, pp. 3820–3860, 2025, doi: 10.1109/COMST.2025.3527641.
- [survey2] "Network Edge Inference for Large Language Models: Principles, Techniques, and Opportunities," *ACM Computing Surveys*, 2026, doi: 10.1145/3809166. (arXiv:2604.22906)
- [llmcache] M. Xu, D. Niyato, and C. G. Brinton, "Serving Long-Context LLMs at the Mobile Edge: Test-Time Reinforcement Learning-based Model Caching and Inference Offloading," *arXiv preprint arXiv:2501.14205*, 2025.
- [slim] "SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference," *arXiv preprint arXiv:2507.06567*, 2025.
- [dnncache] "Joint Optimization of DNN Model Caching and Request Routing in Mobile Edge Computing," *arXiv preprint arXiv:2511.03159*, 2025.

## KV 캐시 / 스케줄링 (기법 배경)

- [kvsurvey] "Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization," *arXiv preprint arXiv:2607.08057*, 2026.
- [kvsched] "Online Scheduling for LLM Inference with KV Cache Constraints," *arXiv preprint arXiv:2502.07115*, 2025.
- [tie] "Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions," *arXiv preprint arXiv:2604.00499*, 2026.
- [mtds] "Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions," *Complex & Intelligent Systems*, 2025, doi: 10.1007/s40747-025-02200-4.
- [orca] G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," in *Proc. USENIX OSDI*, 2022.
- [vllm] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proc. ACM SOSP*, 2023.

## 서비스 마이그레이션 / MEC 오프로딩 (계보)

- [srcl] "Mobility-Aware Seamless Service Migration and Resource Allocation in Multi-Edge IoV Systems," *IEEE Transactions on Mobile Computing*, 2025, doi: 10.1109/TMC.2025.3540407.
- [magrl] "Mobility-Aware Graph Reinforcement Learning for Service Migration in Mobile Edge Computing," in *Proc. CISP-BMEI*, 2024, doi: 10.1109/CISP-BMEI64163.2024.10906084.
- [hybrid] M. Sohaib, S.-W. Jeon, and W. Yu, "Hybrid Online–Offline Learning for Task Offloading in Mobile Edge Computing Systems," *IEEE Transactions on Wireless Communications*, vol. 23, no. 7, pp. 6873–6888, 2024.
- [survey_off] S. Dong et al., "Task Offloading Strategies for Mobile Edge Computing: A Survey," *Computer Networks*, vol. 254, Art. no. 110791, 2024.
