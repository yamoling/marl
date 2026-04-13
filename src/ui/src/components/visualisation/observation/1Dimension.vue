<template>
    <div class="one-d-observation">
        <div class="value-group">
            <h6 class="group-title">Observation</h6>
            <div class="value-grid">
                <span v-for="(o, index) in obs" :key="`obs-${index}`" class="value-pill">
                    {{ o.toFixed(3) }}
                </span>
            </div>
        </div>

        <div class="value-group">
            <h6 class="group-title">Extras</h6>
            <div class="value-grid extras-grid">
                <span v-for="(e, index) in extras" :key="`extra-${index}`" class="value-pill extra-pill">
                    {{ e.toFixed(3) }}
                </span>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">

defineProps<{
    obs: number[],
    extras: number[]
}>();

</script>

<style scoped>
.one-d-observation {
    display: grid;
    gap: 0.65rem;
}

.value-group {
    display: grid;
    gap: 0.3rem;
    text-align: left;
}

.group-title {
    margin: 0;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--bs-secondary-color);
}

.value-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.28rem;
}

.value-pill {
    border: 1px solid var(--bs-border-color);
    background: color-mix(in srgb, var(--bs-secondary-bg) 70%, var(--bs-body-bg));
    border-radius: 999px;
    padding: 0.14rem 0.5rem;
    font-size: 0.76rem;
    font-variant-numeric: tabular-nums;
}

.extra-pill {
    background: color-mix(in srgb, var(--bs-info) 10%, var(--bs-body-bg));
}
</style>