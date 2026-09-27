<script setup lang="ts">
/** Callout shown instead of the replay viewer when the experiment cannot be replayed. */
import type { Issue } from "../../api";
import Icon from "../shell/Icon.vue";

defineProps<{ issue: Issue | null }>();
</script>

<template>
    <div class="callout" role="note">
        <Icon name="film" :size="18" />
        <div>
            <b>Replay isn’t available for this experiment</b>
            <p>
                A replay re-runs the saved policy in the environment, which needs the trainer and the environment to be rebuilt from
                <span class="mono">experiment.json</span>.
            </p>
            <blockquote v-if="issue">
                {{ issue.message
                }}<span v-if="issue.path && !issue.message.includes(issue.path)" class="mono muted"> ({{ issue.path }})</span>
                <pre v-if="issue.detail">{{ issue.detail }}</pre>
            </blockquote>
            <p class="muted">Episode metrics come from the CSV logs and are still accurate.</p>
        </div>
    </div>
</template>

<style scoped>
.callout {
    display: flex;
    gap: 10px;
    border-radius: var(--r-sm);
    padding: 10px 12px;
    font-size: var(--fs-sm);
    background: var(--err-soft);
}
.callout > .icon {
    color: var(--err);
    margin-top: 1px;
}
p {
    margin: 4px 0;
}
blockquote {
    margin: 6px 0;
    padding: 6px 10px;
    border-left: 3px solid var(--err);
    background: color-mix(in srgb, var(--card) 70%, transparent);
    border-radius: 0 6px 6px 0;
}
pre {
    white-space: pre-wrap;
    font-size: var(--fs-xs);
    max-height: 10rem;
    overflow: auto;
    margin: 4px 0 0;
}
</style>
