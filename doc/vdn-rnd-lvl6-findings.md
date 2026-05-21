# Scope and Experimental Setup

This report summarizes the observed exit-rate behavior for VDN-family runs in the lvl6 LLE setting, with emphasis on the RND variants and the layered state-pipeline branches. The evidence covers the baseline VDN configuration, the double-duelling RND family, and the layered IndependentCNN and QCNN branches that appear in the same evaluation slice.

The comparison is intentionally conservative: the available run counts are imbalanced across families, and the main RND comparison is not a clean like-for-like ablation because both the architecture and the state-processing pipeline change between branches. As a result, the observations below should be treated as descriptive rather than causal.

# Result Summary

The baseline VDN remains fixed at an exit_rate of 0.50, which serves as the reference point for the rest of the comparison.

The strongest observed family is VDN-double-duelling-RND-LLE-lvl6-independent, with a median exit_rate of 0.75 and a peak of 1.0. That result is materially above the baseline and indicates that the independent layered branch can reach perfect exit behavior on at least one run.

Other RND-enabled families improve less clearly, and the uneven number of runs makes cross-family ranking unstable. The highest observed peak alone is not sufficient to claim a robust family-wide advantage without matched controls.

# Component Sensitivity

The evidence suggests that the apparent RND gain is sensitive to more than the RND signal itself.

Architecture is a confounder: the strongest family is the independent layered branch, while the compared alternatives include layered CNN variants with different representational capacity. The state pipeline is also a confounder because the layered branches do not share the same input handling as the baseline setup.

Run imbalance further weakens sensitivity claims. A family with fewer or differently distributed runs can look better or worse simply because its observed tail behavior is under-sampled. For that reason, the current evidence supports a hypothesis of interaction effects, not a clean component-level attribution.

# Conclusions and Next Steps

The current data support one cautious conclusion: the layered independent VDN + RND configuration can outperform the baseline VDN exit_rate, but the comparison is not yet controlled enough to isolate the effect of RND from architecture and preprocessing changes.

The next missing controls are matched no-RND versions of the layered IndependentCNN and layered QCNN branches. Those controls are needed to separate the contribution of the layered state pipeline from the contribution of RND itself.

After those controls are added, the most useful follow-up would be a balanced run set across all branches, with the same evaluation budget and the same state-processing path wherever possible. That would make the RND question testable on its own merits.