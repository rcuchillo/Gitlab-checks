Thanks — this is helpful because it shows exactly why the conversation is tense: her self-assessment is largely “delivery + migration work happened”, while her manager’s feedback is “delivery happened but required heavy support, created delays/rework, and there was limited technical independence/initiative”.

Below is a way to use this in tomorrow’s call without it turning into an argument.

1) Start by naming the gap (neutral, evidence-based)

“I’ve read your self-assessment and the manager feedback. They don’t fully align. The purpose of this call is to reconcile the two using specific examples, and then agree a plan that removes ambiguity going forward.”

2) Validate the concern without conceding the rating

“I understand you’re worried about career impact. The most practical way to protect your trajectory is to make the next period very clear: measurable outcomes, a quality bar, and evidence of increasing independence.”

3) Ask for proof points (not opinions)

Use questions that force concrete evidence:

“Which 3 deliverables best demonstrate strong performance on the what? For each: what was delivered, by when, what changed in production/BAU, and where is the evidence (Jira, GitLab MR, run logs, stakeholder confirmation)?”

“On the migrations you listed (EDP→K8, HDFS→bdh1.prod via Starburst): what was your personal contribution vs what was done by others? Which parts did you own end-to-end?”

“Where did you feel blocked (access, unclear requirements, dependencies), and what actions did you take to unblock?”


4) Then anchor on the manager’s specific themes (without loaded language)

Instead of reading the feedback, translate it into testable statements:

“I want to talk through four areas that drove the ‘needs improvement’ view, and check facts.”

1. Independence in Python/EDP/Starburst



“Manager feedback indicates frequent support was needed to execute scripts / convert extraction code and this contributed to delays. Which tasks did you complete independently, and where was support required?”


2. Timeliness / delays



“There were delays cited on Evision 3DS monitoring in Q3’25 and Q4’25. What were the root causes, what was in your control, and what would you do differently?”


3. Quality / rework



“Feedback says monitoring outputs required extensive peer review and there were inefficient ways of working (multiple runs, manual updates). What specific quality issues were raised, and what changes did you implement to prevent recurrence?”


4. Initiative / contribution beyond BAU



“Feedback highlights missed opportunities (reviewing the automated monitoring framework but not proposing enhancements; not progressing DQ framework work; not taking forward CSI enhancement). Which improvement ideas did you propose or try, and what happened?”


5) Use a simple “evidence table” live on the call

If she disputes something, park it into a structure:

Claim (e.g., “deliverables fulfilled as planned”)

Evidence (tickets/MRs/logs)

Impact (reduced manual effort? fewer defects? faster cycle?)

Independence level (owned vs supported)


This keeps the discussion factual and reduces emotion.

6) Close with a forward plan (this is what will calm her down)

Offer a clear 8–12 week plan with measurable expectations aligned to the manager feedback:

Outcomes

Deliver BAU monitoring for X models on time, with agreed definition of done.

Own one enhancement end-to-end (e.g., a small DQ check pack, a CSI improvement, or a Starburst extraction refactor).


Independence

Target: complete weekly run + troubleshooting without escalation except for access issues.

Evidence: runbook updates + GitLab commits/MRs showing changes made independently.


Quality

Reduce peer review cycles: agree “max 1 revision round” target, with a pre-checklist (tests, linting, reproducible run, clear notebook/script separation).


Operating rhythm

15 min weekly checkpoint for blockers + priorities.

Fortnightly demo of improvements (even small ones).


Phrase it like: “I’m not asking for perfection. I’m asking for visible improvement in independence, timeliness, and quality with clear evidence.”

7) If she keeps pushing to change the rating

Keep it calm and process-based: “I’m happy to review any additional evidence you believe materially changes the assessment. The rating itself follows the formal calibration process, so I can’t change it in this call. What I can do is make sure your evidence is considered through the right channel, and make the improvement plan explicit.”


---

If you want, paste the exact wording you plan to use for the first 60 seconds of the call and I’ll tighten it so it sounds firm, fair, and non-cosy.
