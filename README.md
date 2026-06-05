# Predictive Maintenance for Cloud GPU Infrastructure

### A Failure-Mode Profiling Approach Using Unsupervised Learning

When a GPU job fails on a large cloud cluster, somebody has to figure out *why* — and right now that somebody is usually a tired on-call engineer sifting through telemetry logs at 2 a.m. This project automates that triage step. It takes the raw hardware telemetry from a failed job and sorts it into a named, actionable failure category, so infrastructure teams know where to point their attention instead of starting every postmortem from scratch.

---

## 🧩 The Problem (and the Plot Twist)

The original plan was a textbook binary classifier: healthy jobs vs. failed jobs. Then the data had other ideas.

After merging the sensor telemetry with the job-status table, **100% of the surviving records were failures.** Not a sampling fluke — the dataset was architected that way. High-granularity hardware telemetry only got archived for jobs that actually died, presumably to save storage while still leaving enough breadcrumbs for root-cause analysis.

A binary classifier needs both classes to learn anything, so that approach was dead on arrival. The pivot: instead of predicting *whether* a job fails, profile the distinct *ways* GPU jobs fail. That constraint turned out to be the whole project — a model that doesn't just say "it broke," but tells you *how* it broke.

## 🛠️ Skills & Technologies

- **Language:** Python
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Unsupervised Learning:** K-Means (primary), DBSCAN and Isolation Forest (comparison/validation)
- **Feature Engineering:** derived a binary initialization-failure flag, a memory-pressure ratio, a CPU/GPU imbalance metric, and a composite GPU stress score from raw telemetry
- **Workflow:** leakage-safe scaling via `Pipeline`, PCA for visualization, dimensionality reduction, and a rules-based auto-labeler to translate clusters into human-readable categories

## 📊 What the Model Found

K-Means was run with **K=4** — a deliberate override of the silhouette score's preference for K=2. The reasoning: a tidy two-cluster split that lumps every non-initialization failure into one bucket is statistically cleaner but operationally useless. Four named categories that an infrastructure team would actually recognize beat a marginally better silhouette score every time.

The four clusters that fell out of ~6,765 failed jobs across 21 features:

| Cluster | Share | Failure Mode | What It Looks Like |
|--------:|:-----:|:-------------|:-------------------|
| 0 | **79.1%** | Initialization Failure (Primary) | Dead on arrival — never engaged the GPU at all |
| 3 | 17.8% | Mid-Run Memory-Pressure Fault | Started fine, ran out of memory headroom mid-execution |
| 2 | 2.9% | Initialization Failure (Secondary) | A smaller, distinct flavor of init failure |
| 1 | 0.2% | Catastrophic Outlier | 13 records with every metric off the charts — flagged for human review |

**Model quality:** silhouette score of 0.2969 and a Davies-Bouldin index of 1.4482 on the K=4 fit. Both reflect the interpretability-for-cleanliness trade-off baked into the K=4 choice, so they're read as context rather than as a leaderboard score.

**The validation that mattered most was qualitative:** Isolation Forest independently flagged its anomalies almost entirely inside Cluster 1 — the same group the PCA plot and the profile heatmap had already fingered as the weird one. Two unrelated algorithms agreeing that a cluster is the genuine outlier group is a lot more reassuring than any single metric.

## 💡 The Business Takeaway

The headline finding writes its own recommendation: **roughly four out of every five failures happen at initialization, before the job ever does real work.** That means engineering effort spent hardening the job-startup pathway would pay off far more than chasing down rare mid-run faults. It's the difference between fixing the front door everyone trips over and re-engineering a hallway three people a year stub a toe in.

For a cloud provider, that translates directly into uptime, fewer SLA penalties, and a triage process that doesn't depend on which engineer happened to catch the page.

## 🚦 Is It Production-Ready?

Honest answer: not as a standalone system, but yes as a diagnostic decision-support tool. Wired into a cluster-management dashboard alongside the auto-labeler, it can categorize each new failure the moment it happens.

What it **can't** do — by construction — is predict failures before they occur. The training data is 100% failures, so the model has never seen what "healthy" looks like. In production it would either need to be paired with healthy-job telemetry from a complementary source, or scoped strictly as a post-failure diagnostic. Both are legitimate paths; the writeup is upfront about which one this model is built for.

## 📁 What's in Here

- `gpu_failure_mode_profiling.ipynb` — the full notebook: EDA, data prep, feature engineering, modeling, and evaluation, with commentary throughout
- the formal project writeup as a companion PDF

## 📦 Data

[Alibaba GPU Cluster Trace (`cluster-trace-gpu-v2020`)](https://www.kaggle.com/datasets/derrickmwiti/cluster-trace-gpu-v2020) via Kaggle — real-world telemetry from thousands of GPU nodes, widely used as a benchmark for cloud AI infrastructure research. The full sensor table is ~3.8 GB, so the notebook pulls a reproducible 1% random sample (fixed seed) and merges it with the job-status table.

---

*Part of my MS in Data Science & AI coursework. Built end-to-end in Python.*
