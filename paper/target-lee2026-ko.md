# 학습용 한국어 번역: Low-Latency Edge LLM Handover via Joint KV Cache Transfer and Token Prefill

> **원문:** S. Lee, J. Park, C. Zheng, and H. Park, *arXiv:2603.28018*, 2026.
> https://arxiv.org/abs/2603.28018 · https://arxiv.org/pdf/2603.28018
>
> 이 파일은 팀 학습을 위한 **비공식 한국어 번역 노트**입니다. 원문의 대체본이 아니며, 그림은 원문 PDF를 봐야 합니다. 저작권은 원 저자에게 있습니다.

---

**저자:** Seunghun Lee (KAIST), Jihong Park (SUTD, 교신저자), Ce Zheng (Pengcheng Lab), Hyuncheol Park (KAIST, 교신저자)

**핵심어:** Token Streaming, Edge LLM, Key-Value Cache

---

## 초록

대규모 언어 모델(LLM)을 엣지에 두면 대화형 서비스의 지연을 줄일 수 있다. 다만 단말이 기지국(BS) 사이를 핸드오버(HO)하면 서비스가 끊긴다. 대상 쪽 엣지 서버가 디코딩을 바로 이어가려면 단말의 문맥 상태, 즉 KV 캐시를 복구해야 한다. 방법은 두 가지다.

- 토큰을 넘긴 뒤 prefill로 다시 계산한다.
- 백홀로 KV 캐시를 직접 전송한다.

이 논문은 여러 단말에 대해 **prefill 길이와 백홀 KV 전송 스케줄을 함께 정해**, 제일 느린 사용자의 LLM 핸드오버 지연을 최소화하는 통합 설계(ctHO)를 제안한다. 실현 가능 조건이 분명한 단계별 해와, 그에 맞는 전송률 할당 정책을 제시한다. 시뮬레이션에서는 백홀 용량·prefill 속도·문맥 크기를 넓게 바꿔도 제안 방법이 기존 방식보다 나았다.

---

## I. 서론

5G 이후 네트워크에서 LLM 스트리밍은 핵심 응용으로 떠올랐다. ChatGPT, Gemini 같은 서비스를 이미 많은 사람이 모바일로 쓴다. 지금은 대부분 클라우드라서 무선 구간의 지연이 크고 들쭉날쭉하다 [1]. 지연을 줄이고 차별화된 서비스를 주려면 LLM을 네트워크 엣지에 두는 **Edge LLM**이 유력하다 [2], [3].

문제는 이동 중인 사용자다. LLM 추론은 자기회귀라서, 새 토큰은 이전 토큰들의 KV 캐시에 의존한다. 단말이 Edge LLM이 있는 기지국 사이로 넘어가면, **대상 기지국은 그 사용자의 과거 토큰을 디코딩한 적이 없으므로** 대화를 바로 이어갈 수 없다.

단순한 해법은 과거 토큰을 대상 기지국으로 보내 다시 디코딩해 KV 캐시를 만드는 것이다. 이 **토큰 기반 핸드오버(tHO)**의 KV 복원은 LLM의 prefill과 같고, 계산량이 커서 첫 토큰까지 시간(TTFT)이 길어진다. 여러 사용자를 한 배치로 prefill하면, 동시 핸드오버에서 **제일 느린 사용자**가 병목이 된다.

다른 해법은 기지국 사이 백홀로 KV 캐시를 직접 보내는 **캐시 기반 핸드오버(cHO)**다. 다시 디코딩하지 않으니 지연은 줄어들 수 있다. 다만 수십억 파라미터 모델의 KV는 GB 규모일 수 있고, 백홀이 한정되면—특히 핸드오버가 겹치면—전송이 막힌다.

그래서 저자들은 **캐시·토큰을 함께 쓰는 핸드오버(ctHO)**를 제안한다. 일부 KV는 토큰을 받아 prefill로 만들고, 나머지는 백홀로 보내되 **두 일을 동시에** 한다(원문 Fig. 1). 백홀 용량 제약 아래에서 제일 느린 사용자의 핸드오버 지연을 최소화하는 것이 목표다.

**기여 세 가지**

1. prefill로 만들 KV 길이와 백홀 전송률 할당을 같이 최적화해 Edge LLM 핸드오버 지연을 줄이는 ctHO.
2. 주어진 prefill 길이에 대해 백홀 스케줄을 정한 다음 prefill 길이를 고르는 2단계 최적화, 그리고 전역 최적 증명.
3. 시뮬레이션에서 최악 사용자 핸드오버 지연이 cHO 대비 최대 약 3.1배, tHO 대비 최대 약 3.7배 감소.

토큰 통신은 다중접속 [4], 패킷화 [5], 멀티모달 전송 [6]에서 연구됐지만, 기존 연구는 정지한 사용자를 가정한다. 이 논문은 **이동 사용자의 핸드오버**에 초점을 둔다. KV 압축 [7]이나 에이전트 간 캐시 통신 [8], [9]은 이 설계와 직교하므로, 같이 쓰면 지연을 더 줄일 수 있다.

---

## II. 시스템 모델과 문제 정의

### A. Edge LLM 토큰 스트리밍 중의 다중 단말 핸드오버

출발 기지국이 $K$명의 단말에 LLM 토큰을 각각 스트리밍한다. 단말들은 대상 기지국 쪽으로 움직이고, 단말 $i$는 시각 $\tau_i$에 핸드오버를 시작한다. 두 기지국은 같은 LLM 구조·파라미터를 쓴다.

핸드오버가 일어나면 스트리밍은 멈추고, 대상 기지국은 그때까지 디코딩된 토큰 수 $C_i$에 해당하는 KV 캐시를 준비한다. KV가 준비된 뒤에야 스트리밍이 재개되며, 그 사이 시간이 **최소화할 Edge LLM 핸드오버 지연**이다.

대상 기지국은 출발 기지국에서 토큰을 받아 일부 KV를 prefill로 만들고, 나머지는 백홀로 받는다 [7]. 토큰 자체를 보내는 지연은 KV 전송·prefill보다 무시할 수 있다고 가정한다. 스트리밍은 **둘 다 끝난 뒤에만** 재개되므로, 핸드오버 지연은 두 지연 중 **더 큰 쪽**이 된다.

따라서 (1) prefill로 처리할 KV 양과 (2) 단말별 백홀 용량 할당을 함께 정한다. 출발 기지국은 $\{(\tau_i, C_i)\}_{i=1}^{K}$와 대상 기지국의 prefill 지연 특성을 안다고 가정한다. 일반성을 잃지 않고 핸드오버 시각 순으로 정렬한다.

$$
\tau_1 \le \tau_2 \le \cdots \le \tau_K. \tag{1}
$$

대상 기지국에서 사용자마다 따로 prefill하면 느리다. 그래서 **배치 prefill**으로 여러 단말의 토큰을 한 번에 처리한다 [10]. 모든 단말에 공통 접두 길이 $L$만큼 KV를 만들며, $L \in [0, C_{\max}]$, $C_{\max} \triangleq \max_i C_i$이다. 배치 입력 길이는 같아야 하므로 $C_i < L$이면 있는 $C_i$개만 쓰고 나머지는 제로 패딩한다. 단말 $i$의 $C_i$개 토큰은 이렇게 나뉜다.

$$
n_i^{(\mathrm{pf})}(L) \triangleq \min\{C_i, L\}, \tag{2}
$$

$$
n_i^{(\mathrm{tx})}(L) \triangleq C_i - n_i^{(\mathrm{pf})}(L) = (C_i - L)^+, \tag{3}
$$

여기서 $(\cdot)^+ = \max\{\cdot, 0\}$이다. $n_i^{(\mathrm{pf})}$는 prefill로 계산하는 토큰 수, $n_i^{(\mathrm{tx})}$는 백홀로 KV를 보내는 나머지 토큰 수다.

시각 $t$에 단말 $i$에 할당하는 백홀 전송률 $r_i(t)$는 용량 제약을 따른다.

$$
\sum_{i=1}^{K} r_i(t) \le R \quad [\text{tokens/s}], \quad \forall t. \tag{4}
$$

$R \triangleq R_{\mathrm{bh}} / s_{\mathrm{KV}}$는 토큰당 KV 페이로드 크기 $s_{\mathrm{KV}}$로 정규화한 백홀 용량이다. $R$과 $r_i(t)$를 tokens/s로 두면 $L$, $C_i$와 단위가 맞는다.

### B. 최악 사용자 LLM 핸드오버 지연 최소화

배치 prefill 완료 시각을 $T^{(\mathrm{pf})}(L)$, 단말 $i$의 캐시 전송 완료 시각을 $T_i^{(\mathrm{tx})}(L, r_i, C_i)$라 하면, 핸드오버 이후 각각의 지연은

$$
D_i^{(\mathrm{pf})}(L) \triangleq T^{(\mathrm{pf})}(L) - \tau_i, \tag{5}
$$

$$
D_i^{(\mathrm{tx})}(L, r_i, C_i) \triangleq T_i^{(\mathrm{tx})}(L, r_i, C_i) - \tau_i. \tag{6}
$$

디코딩은 둘 다 끝나 KV가 만들어진 뒤에만 재개되므로, 단말 $i$의 LLM 핸드오버 지연은

$$
D_i(L, r_i, C_i) \triangleq \max\bigl\{ D_i^{(\mathrm{pf})}(L),\; D_i^{(\mathrm{tx})}(L, r_i, C_i) \bigr\}. \tag{7}
$$

최악 사용자 지연은

$$
D(L, r) \triangleq \max_{i \in \{1,\ldots,K\}} D_i(L, r_i, C_i). \tag{8}
$$

배치 prefill 주기 간격을 $T_c$라 하자. 배치는 가장 늦은 핸드오버 $\tau_K$ 이후 첫 주기 경계에서 시작한다.

$$
t_s \triangleq \Bigl\lceil \frac{\tau_K}{T_c} \Bigr\rceil T_c.
$$

prefill 완료 시각은

$$
T^{(\mathrm{pf})}(L) \triangleq t_s + p(L), \tag{9}
$$

여기서 $p(L)$은 배치 prefill 지연이다. 캐시 전송 완료 시각은 나머지 $n_i^{(\mathrm{tx})}(L)$ 토큰분 KV가 다 도착한 가장 이른 시각이다.

$$
T_i^{(\mathrm{tx})}(L, r_i, C_i) \triangleq \inf\left\{ t \ge \tau_i : \int_{\tau_i}^{t} r_i(u)\,du \ge n_i^{(\mathrm{tx})}(L) \right\}. \tag{10}
$$

최적화 문제 $\mathrm{P}$는 공통 prefill 길이와 백홀 전송률을 골라 최악 사용자 지연을 최소화하는 것이다.

$$
\mathrm{P}:\quad \min_{0 \le L \le C_{\max},\; r(\cdot)} D(L, r) \tag{11a}
$$

$$
\text{s.t.}\quad \sum_i r_i(t) \le R. \tag{11b}
$$

$L$과 $r(\cdot)$을 같이 최적화하지만, **단계적으로 풀어도 최적이 깨지지 않는다.** 고정된 $L$에 대해 $r(\cdot)$을 먼저 최적화하고, 그다음 $L$을 고른다. 가치 함수를

$$
V(L) \triangleq \min_{r(\cdot)} D(L, r) \tag{12}
$$

로 두고, 주어진 $L$에서 최적인 전송률 정책을 $r^\star(L)$, $L^\star = \arg\min_{0\le L\le C_{\max}} V(L)$이라 하자.

**명제 1.** 쌍 $(L^\star, r^\star(\cdot))$은 $\mathrm{P}$의 전역 최적해다.

*증명 요지.* 임의의 가능해 $(L, r)$에 대해 정의상 $V(L) \le D(L, r)$이다. 또 $D(L^\star, r^\star(L^\star)) = V(L^\star)$이고 $L^\star$의 최적성 때문에 $V(L^\star) \le V(L)$이다. 따라서

$$
D(L^\star, r^\star(L^\star)) = V(L^\star) \le V(L) \le D(L, r). \tag{13}
$$

---

## III. 백홀 스케줄과 prefill 길이의 단계적 최적화

명제 1에 따라 $\mathrm{P}$를 두 부분 문제로 나눈다.

$$
\mathrm{P1}:\quad V(L) := \min_{r(\cdot)} D(L, r) \quad (L \text{ 고정}), \tag{14}
$$

$$
\mathrm{P2}:\quad \min_{0 \le L \le C_{\max}} V(L). \tag{15}
$$

먼저 고정 $L$에서 $\mathrm{P1}$을 풀어 $V(L)$을 구한 뒤, $\mathrm{P2}$로 최적 $L$을 고른다.

### A. 최적 캐시 전송 지연과 prefill 길이

고정 $L$에서 prefill 지연은 $r(\cdot)$과 무관하다. 그래서 $\mathrm{P1}$은 최악 사용자 **캐시 전송 지연** $D^{(\mathrm{tx})}(L, r)$을 최소화하는 문제로 줄어든다.

$$
\begin{aligned}
V(L)
&= \min_{r(\cdot)} D(L, r) \\
&= \min_{r(\cdot)} \max_i \max\bigl\{ D_i^{(\mathrm{pf})}(L),\; D_i^{(\mathrm{tx})}(L, r_i, C_i) \bigr\} \\
&= \max\Bigl\{ D^{(\mathrm{pf})}(L),\; \min_{r(\cdot)} D^{(\mathrm{tx})}(L, r) \Bigr\},
\end{aligned} \tag{16}
$$

여기서 $D^{(\mathrm{pf})}(L) \triangleq \max_i D_i^{(\mathrm{pf})}(L)$, $D^{(\mathrm{tx})}(L, r) \triangleq \max_i D_i^{(\mathrm{tx})}(L, r_i, C_i)$이다.

앞쪽 $k$명에게 남은 KV의 누적량을

$$
S_k(L) \triangleq \sum_{i=1}^{k} n_i^{(\mathrm{tx})}(L), \quad k \in \{1,\ldots,K\} \tag{17}
$$

로 두면, 고정 $L$에서 달성 가능한 최소 캐시 전송 지연은 다음과 같다.

**명제 2.** 임의의 고정 $L \in [0, C_{\max}]$에 대해

$$
D^\star_{(\mathrm{tx})}(L) \triangleq \min_{r(\cdot)} D^{(\mathrm{tx})}(L, r)
= \max_{k \in \{1,\ldots,K\}} \left[ \frac{S_k(L)}{R} - (\tau_k - \tau_1) \right]^+. \tag{18}
$$

*증명 요지.* 어떤 가능 할당 $r(\cdot)$에서도 모든 단말의 전송은 $\tau_i + D^{(\mathrm{tx})}(L, r)$ 안에 끝나야 한다. KV는 $\tau_i$ 이후에만 보낼 수 있으므로, 시각 $\tau_k + D^{(\mathrm{tx})}$까지는 앞 $k$명 몫 $S_k(L)$을 보내야 한다. 반면 용량 $R$로는 구간 $[\tau_1,\; \tau_k + D^{(\mathrm{tx})}]$ 동안 최대 $R\bigl(D^{(\mathrm{tx})} + \tau_k - \tau_1\bigr)$ 토큰분만 보낼 수 있다. 정리하면 모든 $k$에 대해

$$
D^{(\mathrm{tx})}(L, r) \ge \left[ \frac{S_k(L)}{R} - (\tau_k - \tau_1) \right]^+. \tag{21}
$$

따라서 최소값도 이 하한들의 최댓값 이상이다. 이 하한을 실제로 달성하는 정책은 III-B절에 있다.

배치 prefill이 끝나기 전에 나머지 KV를 이미 다 보낼 수 있는 단말은, 그 단말만 보면 배치 prefill이 필요 없다. 이후 논의는 그렇지 않은 단말에 초점을 둔다.

이제 $\mathrm{P2}$에서 $L$을 고른다. $D^{(\mathrm{pf})}(L)$은 $L$이 커질수록 늘고(비감소), $D^\star_{(\mathrm{tx})}(L)$은 $L$이 커질수록 준다(비증가). 가능하면 **두 지연이 같아지는 $L$**에서 최솟값이 나온다. 어느 한쪽이 전체를 지배하지 않게 맞추는 것이다(원문 Fig. 2).

**명제 3.** $[0, C_{\max}]$ 안에 $D^{(\mathrm{pf})}(L) = D^\star_{(\mathrm{tx})}(L)$인 $L$이 있으면, 그런 $L$은 $V(L)$을 최소화한다. 없으면 최적은 경계, 즉 $L^\star \in \{0, C_{\max}\}$에 있다.

*증명 요지.* $f(L) = D^{(\mathrm{pf})}(L)$ (비감소), $g(L) = D^\star_{(\mathrm{tx})}(L)$ (비증가)이고 $V(L) = \max\{f(L), g(L)\}$이다. $f(L_0)=g(L_0)$이면 $L < L_0$에서는 $V=g \ge g(L_0)$, $L > L_0$에서는 $V=f \ge f(L_0)$이므로 $L_0$이 최소다. 교점이 없으면 전 구간에서 $f>g$이거나 $f<g$이고, 최솟값은 각각 $L=0$ 또는 $L=C_{\max}$에서 나온다.

### B. 최적 백홀 전송률 스케줄

명제 2는 최적값 $D^\star_{(\mathrm{tx})}(L)$만 알려 준다. $\mathrm{P1}$을 끝내려면 그 값을 **실제로 만드는** $r(\cdot)$이 필요하다. 저자들은 한 번에 단말 하나에게만 백홀을 주는 단순한 정책을 쓴다.

시각 $t$에 백홀을 받는 단말 번호를 $\pi(t)$라 하자. 스케줄러는 **활성 집합이 바뀔 때만** 갱신된다. 새 단말이 $\tau_i$에 들어오거나, 지금 보내던 단말의 나머지 KV가 다 나갔을 때다.

시각 $t$에 단말 $i$의 남은 토큰 양은

$$
n_i^{(\mathrm{rem})}(t, L) \triangleq \left[ n_i^{(\mathrm{tx})}(L) - \int_{\tau_i}^{t} r_i(s)\,ds \right]^+.
$$

활성 집합은 핸드오버는 이미 일어났는데 KV 전송은 아직 안 끝난 단말들이다.

$$
A(t) \triangleq \bigl\{ i \in \{1,\ldots,K\} : t \ge \tau_i,\; n_i^{(\mathrm{rem})}(t, L) > 0 \bigr\}. \tag{23}
$$

단말 $i$의 목표 전송 완료 시각을 $d_i(L) \triangleq \tau_i + D^\star_{(\mathrm{tx})}(L)$로 둔다. 매 시각 $t$에 $A(t)$ 안에서 **마감이 가장 이른** 단말에게 백홀 전체를 준다.

$$
\pi(t) = \arg\min_{i \in A(t)} d_i(L), \tag{24}
$$

$$
r_i(t) = R \cdot \mathbf{1}\{\pi(t) = i\}. \tag{25}
$$

백홀은 한 단말에만 몰아 주고, $\pi(t)$는 $A(t)$가 바뀔 때만 바뀐다. $A(t)$가 비어 있지 않으면 백홀은 놀지 않는다. 시각 $\tau_k + D^\star_{(\mathrm{tx})}(L)$까지 이 정책이 보낼 수 있는 양은 각 $k$의 $S_k(L)$을 채우기에 충분하다. 이것으로 고정 $L$에서 $D^\star_{(\mathrm{tx})}(L)$을 달성하는 $r(\cdot)$이 나오고, $\mathrm{P1}$이 끝난다.

한 줄로 말하면, **마감이 빠른 단말부터 백홀을 독점 할당**하는 방식이다.

---

## IV. 시뮬레이션 결과

### A. 실험 설정

- 단말 수: 기본 $K=4$.
- 문맥 길이: 각 $C_i$를 $\mathrm{Unif}[1024, C_{\max}]$에서 뽑음.
- 배치 prefill 주기: $T_c = 0.01\,\mathrm{s}$.
- 지형: 1차원 직선. 출발 기지국 $x^{(s)}=0$, 대상 기지국 $x^{(t)}=D_{\mathrm{bs}}=300\,\mathrm{m}$ [11].
- 이동: 단말은 $x_i(0) \sim \mathrm{Unif}[120,130]\,\mathrm{m}$에서 출발, 속력 $v_i=20\,\mathrm{m/s}$, $x_i(t)=x_i(0)+v_i t$.
- 핸드오버: 경계 $x_b=150\,\mathrm{m}$를 지날 때. $\tau_i = (x_b - x_i(0))/v_i$.
- 해석: 도로를 따라 차가 움직이며 길가 접속점에 순서대로 붙는 상황을 단순화한 모델.

토큰당 KV 크기:

$$
s_{\mathrm{KV}} = 2 \cdot N_\ell \cdot N_{\mathrm{kv}} \cdot d_h \cdot q \quad \text{[bits]}.
$$

앞의 2는 key/value, $N_\ell$은 레이어 수, $N_{\mathrm{kv}}$는 KV 헤드 수, $d_h$는 헤드 차원, $q$는 원소당 비트다. 모델은 **Qwen2.5-7B-Instruct**, $(N_\ell, N_{\mathrm{kv}}, d_h)=(28, 4, 128)$, $q=16$ bit. 그러면 $s_{\mathrm{KV}}=458{,}752$ bit/token이고, $C=3072$ 토큰이면 KV 전체는 약 **176 MB**.

prefill 지연 모델은 $p(L)=aL+b$이며, 기본값 $a=9.4267\times 10^{-5}$, $b=2.4\times 10^{-3}$이다.

### B. 백홀 속도, prefill 속도, 캐시 크기, 단말 수의 영향

비교 대상은 tHO와 cHO다.

$$
\text{tHO:}\quad L=C_{\max} \;\Rightarrow\; n_i^{(\mathrm{pf})}=C_i,\; n_i^{(\mathrm{tx})}=0, \tag{26}
$$

$$
\text{cHO:}\quad L=0 \;\Rightarrow\; n_i^{(\mathrm{pf})}=0,\; n_i^{(\mathrm{tx})}=C_i. \tag{27}
$$

tHO는 전부 배치 prefill, cHO는 전부 백홀 전송이다. ctHO는 $L$을 그 사이에 둔다. 아래는 원문 Fig. 3의 요지다.

**(a) 백홀 속도 $R_{\mathrm{bh}}$.** tHO는 백홀을 안 쓰므로 곡선이 안 변한다. cHO는 $R_{\mathrm{bh}}$가 커질수록 줄고, 백홀이 아주 빠르면 ctHO에 가까워진다. $R_{\mathrm{bh}}=2\,\mathrm{Gbps}$에서는 cHO 지연이 ctHO보다 **3.1배 이상** 크다.

**(b) prefill 속도 $1/a$** ($R_{\mathrm{bh}}=4.5\,\mathrm{Gbps}$). cHO는 prefill을 안 쓰므로 안 변한다. tHO는 prefill이 빨라질수록(작은 $a$) 좋아진다. ctHO는 계산이 느리면 $L^\star$를 줄여 백홀에 더 기대고, 계산이 빠르면 $L^\star$를 키워 백홀 부하를 줄인다. 전 구간에서 지연이 가장 작다.

**(c) 최대 캐시 $C_{\max}$.** 문맥이 길수록 재개 전 계산·전송이 늘므로 최악 지연이 커진다. 전 스윕에서 ctHO가 가장 낮다.

**(d) 단말 수 $K$.** 단말이 늘면 같은 백홀·prefill을 나눠 쓰므로 세 방법 모두 지연이 커진다. 격차는 $K$가 클수록 커진다. 예: $K=12$에서 ctHO 약 **0.95 s**, tHO **1.25 s**, cHO **3.35 s**.

### C. 핸드오버 있는 경우와 없는 경우의 총 스트리밍 지연

핸드오버 지연만이 아니라, 핸드오버 **이후** 새로 만든 $G=1024$ 토큰을 무선으로 보내는 시간까지 합친 **최악 사용자 총 스트리밍 지연**을 $D_{\mathrm{bs}}$를 바꿔 가며 본다(원문 Fig. 4).

단말 $i$의 SNR은 서빙 기지국과의 거리 함수다. 핸드오버 없음은 출발 기지국 $x^{(s)}=0$, 핸드오버 있음은 대상 기지국 $x^{(t)}=D_{\mathrm{bs}}$이다.

$$
\gamma_i^{(b)}(t) = \gamma_{\mathrm{ref}} \left( \frac{d_{\mathrm{ref}}}{|x_i(t)-x^{(b)}|} \right)^{\alpha}, \tag{28}
$$

$(\gamma_{\mathrm{ref}}, d_{\mathrm{ref}}, \alpha)=(10\,\mathrm{dB},\; 20\,\mathrm{m},\; 3.5)$. 무선 토큰 전송률은

$$
r^{(b)}(t) \triangleq \frac{W \log_2\bigl(1+\gamma_i^{(b)}(t)\bigr)}{s_{\mathrm{tok}}} \quad [\text{tokens/s}], \tag{29}
$$

$s_{\mathrm{tok}}=12$ bits/token, $W=2\,\mathrm{MHz}$.

$\tau_i$부터 잰 총 지연은 핸드오버 지연과 이후 스트리밍 지연의 합이다. 핸드오버가 없으면 핸드오버 항은 0이고 무선 스트리밍만 남는다.

$D_{\mathrm{bs}}$가 커질수록, 출발 기지국에 붙어 있는(핸드오버 없음) 링크의 SNR이 급격히 나빠져 지연이 빨리 늘어난다. $D_{\mathrm{bs}} < 375\,\mathrm{m}$에서는 핸드오버 비용이 없어서 **핸드오버 없음**이 더 낫다. 거리가 더 벌어지면 핸드오버하는 쪽이 이기고, $D_{\mathrm{bs}}=500\,\mathrm{m}$에서는 핸드오버 없음이 ctHO보다 **2.1 s 이상** 느리다. 핸드오버를 하는 방법들 사이에서는 전 구간에서 ctHO가 가장 작다.

---

## V. 결론

이 논문은 여러 단말이 토큰 스트리밍 중에 움직일 때, 대상 기지국이 KV 캐시를 어떻게 복구할지를 다룬다. 제안하는 **ctHO**는 prefill 양과 백홀 KV 전송률을 함께 정하고, prefill 길이와 백홀 스케줄을 단계적으로 풀어 최악 사용자 핸드오버 지연을 최소화한다. 실험 설정 안에서는 기존 tHO·cHO보다 일관되게 나았다.

향후 과제는 지금처럼 핸드오버 순간에만 복구하는 **hard HO**를, 대상 기지국이 미리 계산하고 출발 기지국은 디코딩을 이어 가는 **soft HO**로 확장해 끊김을 더 줄이는 것이다.

---

## 이 저장소와의 관계 (학습 메모)

시뮬레이터의 `reactive-hybrid`가 이 논문의 ctHO를 **단순화한 근사**다. 핸드오버가 **이미 일어난 뒤**에 전송/재계산을 나눠 지연의 최댓값을 줄인다. 이 논문의 목적 함수는 평균이 아니라 **최악 사용자 지연**이고, 백홀은 여러 단말에 대해 마감 우선으로 스케줄한다.

우리 초안의 주장은, 이 반응형 최적도 원리적으로는 핸드오버 **이전**의 선제 전송(시간축 혼잡 평탄화)을 할 수 없다는 점이다. 원문 결론의 future work(soft HO, 대상 측 사전 계산)와도 맞닿아 있다.

---

## 참고문헌

원문 목록은 PDF 5쪽 또는 https://arxiv.org/pdf/2603.28018 을 따른다. 이 저장소 초안에서의 인용 키는 `paper/references.md`의 `(키: target)`이다.
