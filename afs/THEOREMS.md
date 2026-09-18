# Theorems for the Associative Factor Store
(All proofs self-contained; empirical anchors marked [MEASURED].)

Notation: model width d; E table rows; per-row trained corrections of size c
(params); top-k retrieval with width k; bytes/param b (bf16: 2, FP8: 1, FP4: 0.5);
ridge point R* (flop/byte at which a kernel becomes compute-bound; H200 bf16 ≈ 206).
Knowledge-capacity law: a trained parameter stores ≤ 2 bits of extractable
knowledge at ≥1,000 exposures (Allen-Zhu & Li, arXiv 2404.05405) [MEASURED, cited].

---------------------------------------------------------------- T1 (capacity)
**Theorem.** The AFS store of E rows with c trained correction-params per row
has knowledge capacity at most 2·E·c bits, achieved when every fact is exposed
≥1,000 times and routed consistently.
*Proof.* The store's trained parameters are exactly the union of the per-row
corrections (keys contribute E·d params, charged separately in T4) and the
frozen basis carries no learned bits (never updated). By the capacity law each
trained param stores ≤ 2 bits of *extractable* knowledge; summing over rows
gives the bound. Achievement requires (i) the exposure condition per fact and
(ii) routing consistency (T6). ∎
Corollary (design target): dense-1B knowledge (707M FFN params) ⇔ E·c = 707M.
The worked config: E=320 rows × c=2.36M (rank-128 corrections) = 755M ≥ 707M. ✓

--------------------------------------------------- T2 (widening exactness)
**Theorem.** The zero-padded rank-widening map φ_{r→R} applied to U∈R^{d_out×r},
V∈R^{r×d_in} preserves the operator exactly: for all x,
  x·(V_φᵀ U_φᵀ) = x·(Vᵀ Uᵀ).
*Proof.* V_φ = [V; 0] (block column), U_φ = [U; 0] (block row). Then
V_φᵀ U_φᵀ = [Vᵀ 0]·[Uᵀ; 0] = VᵀUᵀ + 0-matrices = VᵀUᵀ, since every product
term with a zero block is the zero matrix and block-wise addition of zero
matrices is exact in any IEEE-754 rounding mode. The same argument holds
per-accumulation in a matmul kernel: terms with a zero factor contribute
+0.0, and x+0.0 = x exactly (IEEE 754, incl. signed zeros and subnormals). ∎
[MEASURED] fp64 forward diff = 7.99e-15 (fp64 reassociation floor); bf16 diff
1.22 on logits of magnitude ~64-128 = bf16 ULP noise (eps(128)=1.0), the same
tolerance class the fast-path docstring already accepts.

------------------------------------------- T3 (traffic–capacity decoupling)
**Theorem.** For flat keys, per-token weight traffic is
  T = E·d·b_K + k·c·b_W + O(activations),
so the *marginal* bytes per marginal bit of knowledge capacity is
  ∂T/∂(2Ec) = d·b_K/(2c)  →  0  as c grows.
With two-level product keys (T4), ∂T/∂(2Ec) = d·b_K·(1/E1+1/E2)/(2c) → 0 doubly.
*Proof.* Scoring reads all E keys (E·d values, b_K bytes each); retrieval reads
only the k selected rows' corrections (k·c·b_W). Differentiating T in E at fixed
k, c, d isolates the key term. Capacity differentiates in E·c. The ratio is the
marginal bandwidth price of knowledge; it vanishes whenever c ≫ d·b_K/(2b_W),
which holds at c = 2.36M ≫ d = 2048 (ratio ≈ 1/2,300). ∎
**Interpretation.** Knowledge growth costs bytes only through key-scoring; the
decoupling factor is c/d ≈ 1,100 for the worked config, and polynomially better
with product keys. This is the formal statement of "not bandwidth limited":
the bandwidth bill of *what the model knows* grows sublinearly and → 0/param.

--------------------------------------- T4 (product-key sublinear scoring)
**Theorem.** With product keys — row (i,j) keyed by k1_i + k2_j, i∈[E1], j∈[E2] —
the score of row (i,j) is s_ij = (x·k1_i + x·k2_j)/√d, so computing all E1·E2
scores costs (E1+E2)·d MACs instead of E1·E2·d.
*Proof.* Linearity of the dot product: s_ij = x·(k1_i+k2_j)/√d = (s1_i+s2_j)/√d
with s1 = x·k1ᵀ ∈ R^{E1}, s2 = x·k2ᵀ ∈ R^{E2}. The score matrix is the rank-1
sum s1 ⊕ s2, computable from the two vectors. ∎
**Corollary.** Effective table size E = E1·E2 with scoring cost O(E1+E2)·d:
capacity scales *multiplicatively* while scoring scales *additively* — the
key-traffic term of T3 becomes vanishingly sublinear in capacity.

------------------------------------------------ T5 (compute-bound crossover)
**Theorem.** Serving a batch of B tokens is compute-bound iff
  B ≥ R*·b_W/2.
*Proof.* Per forward: FLOPs = 2·k·c·B (the selected rows' GEMMs), traffic =
k·c·b_W + E·d·b_K. Ignoring the key term (T3), arithmetic intensity =
2B/b_W. Compute-bound iff 2B/b_W ≥ R*, i.e. B ≥ R*·b_W/2. ∎
[MEASURED anchor] bf16, H200: B* ≈ 206·2/2 = 103 ≈ the ~100 crossover stated;
FP8 b_W=1 halves it to ~52; FP4 to ~26. Every real multi-user server exceeds
B*; hence the AFS serving regime is compute-bound in the selected rows'
GEMMs — and their FLOPs are capacity-independent (k, c fixed), so *serving
throughput is invariant to how much the model knows*.

------------------------------------- T6 (exposure–routing risk, quantified)
**Theorem.** If a fact's tokens route to its storing row with probability p,
the achieved capacity per that row's parameters is
  2 bits/param · g(p·n_exp/1000),   g(x) = min(1, x)  (sigmoidal near 1).
*Proof.* Direct from the exposure law (capacity is exposure-limited below
~1,000 exposures/fact, Allen-Zhu Result 4) applied to the per-row effective
exposure n_eff = p·n_exp: achieved = 2·min(1, n_eff/1000) bits/param. ∎
**Risk statement.** Unlike dense training (all params exposed to every token),
AFS concentrates a fact's exposure on its routed row: routing inconsistency
multiplies the exposure deficit. The probe logs per-row routing histograms;
the capacity instrument measures g directly. Mitigations: soft retrieval
(all rows get gradient, weighted), gate-entropy regularization, key warmup.

-------------------------------- T7 (optical–electronic function equivalence)
**Theorem.** The optical rendering (all E plates illuminated in parallel,
photocurrents weighted by the soft scores w) computes exactly the electronic
soft-mode function Σ_k w_k·(P_k·x).
*Proof.* Photodetection is linear in optical power: the summed current from
plate k weighted by gate w_k contributes w_k·(P_k·x); summing currents adds
the terms. Weighted summation commutes with the linear operators: the total
equals (Σ_k w_k P_k)x, which is the electronic soft-mode output. ∎
**Corollary (hard-mode gap).** The electronic top-k mode computes
Σ_{k∈topk} w_k·(P_k x) ≠ soft mode unless the excluded weights are ~0.
The gap ||soft−hard|| is a measurable training quantity (probe); gate-entropy
regularization minimizes it so the optical (soft) and electronic (hard)
renderings converge to the same function.
**Note (no gating hardware needed).** Because the optical path computes all
plates in parallel for free, the optical deployment runs the SOFT mode natively
— no mask SLM required; the weights come from electronic scoring.

------------------------------------------------------ T8 (scaling to targets)
**Theorem.** For a knowledge target of P trained FFN params, any (E, c) with
E·c = P meets the capacity bound; the optimal row count minimizes total
traffic T(E) = E·d·b_K + k·(P/E)·b_W, giving
  E* = sqrt(k·P·b_W / (d·b_K))  (AM-GM).
*Proof.* T(E) is convex in E; setting dT/dE = d·b_K − k·P·b_W/E² = 0 gives E*.
At the optimum, both key-scoring and row-fetch traffic are equal. ∎
[MEASURED config check] P = 707M, k = 4, b = 1 (FP8): E* = sqrt(4·707e6/2048)
≈ 37,000 rows × 19K params/row — the product-key structure (T4) is what makes
~37k-row tables scoreable: two 192-dim key sets → scoring 384 dot products
for a 36,864-row table. Flat keys at E=320 were the suboptimal point of the
first worked config; product keys are the scaling-correct design.
