You’re right to be cautious. **Screening** (sanctions/PEP/adverse media/name matching) and **transaction monitoring (TM)** share “case management,” but the *hard parts* are different. If SymphonyAI’s strongest published numbers are screening-led, here are the main limitations/risk areas when applying similar “98% agreement / 90% effort reduction” expectations to TM.

## Where screening results don’t translate cleanly to TM

### 1) Ground truth is weaker in TM

* **Screening** often has clearer adjudication outcomes (match / no match) and external reference lists.
* **TM** is about *suspicion* and typology interpretation; SAR filing is a delayed/partial label and often inconsistent across investigators.

**Implication:** “Agreement” may drop or become less meaningful unless you define a robust reference standard (QA panel, typology rubric, outcome proxies).

### 2) Much higher context and narrative complexity

TM cases typically require:

* multi-transaction patterns over time,
* counterparty networks,
* customer behaviour baselines,
* product/channel context,
* and rationale for why it breaches typology thresholds.

**Implication:** Agents need stronger tooling (graph, aggregations, peer groups, sequence features). Without that, you can get fluent narratives that are incomplete or mis-specified.

### 3) Tooling dependency is heavier

To be credible in TM, an “agent” must reliably:

* pull the right transaction subsets,
* compute aggregates (velocity, round-tripping, cash intensity),
* join KYC/EDD, account linkages, device/IP, merchant data,
* and cite evidence.

**Implication:** Performance becomes **integration-limited**. If data access is slow/fragmented or calculations aren’t standardized, productivity gains shrink and variance increases.

### 4) Alert volumes and heterogeneity are bigger

TM queues span many typologies (mules, structuring, laundering, scams, trade, crypto rails, etc.) and business lines. Screening workflows are often more uniform.

**Implication:** A single “% effort reduction” headline is unlikely to hold across typologies. You’ll see a wide spread: some typologies automate well; others don’t.

### 5) False positive reduction is not the same lever

In screening, big gains come from discounting obvious non-matches at scale.
In TM, big gains usually come from:

* better alert quality upstream (scenario tuning/modeling),
* alert grouping/roll-up,
* and better triage prioritisation.

**Implication:** An “agent” alone may reduce *handling time*, but may not materially reduce *alert volume* unless combined with detection changes.

### 6) Higher model risk / governance burden

TM decisions are typically more scrutinised: explainability, traceability, and consistency with internal policy and regulatory expectations.

**Implication:** You’ll need stronger controls:

* evidence citations for every claim,
* “abstain/escalate” rules,
* QA sampling,
* and audit logs of every tool call and source.

### 7) Higher hallucination risk impact

If a screening agent hallucinates, it’s often caught by match logic or list checks.
If a TM narrative hallucinates transactions/relationships, it can create **unsupported suspicion** or **miss key suspicion**.

**Implication:** You need hard guardrails: retrieval-only generation, transaction IDs in output, and automatic cross-checks (e.g., “every amount/date in narrative must map to a transaction record”).

### 8) Investigator workflow differs

Screening often has short, repetitive steps; TM investigations can involve RFIs, customer contact outcomes, and multi-stage approvals.

**Implication:** “Manual effort” in TM is not just review time; it includes waiting, coordination, and documentation—so the measurable uplift may be smaller unless you automate the full chain.

---

## How to frame a TM pilot so you don’t get misled by screening-style metrics

Instead of asking for “90% effort reduction,” measure by TM-relevant KPIs:

* **AHT per alert** by typology (median + distribution)
* **% cases closed at L1 vs escalated** (and *why*)
* **QA defect rate** (missing evidence, wrong typology mapping, unsupported statements)
* **Narrative edit time** (minutes saved, not just “quality”)
* **Evidence traceability rate** (% narrative claims with linked transactions/KYC sources)
* **Abstention/escalation rate** (healthy guardrail behaviour)

If you want, I can translate this into a **TM-specific pilot scorecard** with “what good looks like” thresholds (e.g., minimum traceability, maximum defect rate, target AHT reduction) so you can compare vendors fairly and avoid over-indexing on screening numbers.
