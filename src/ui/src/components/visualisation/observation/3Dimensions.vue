<template>
    <div class="observation-preview">
        <h6 class="section-title">Layers</h6>
        <div class="layers-wrap">
            <table v-for="(layer, layerIndex) in obs" :key="layerIndex" :style="{ width: `${layer.length * 10}px` }"
                class="layer-table">
                <tbody>
                    <tr v-for="(row, rowIndex) in layer" :key="rowIndex">
                        <td v-for="(item, cellIndex) in row" :key="cellIndex" class="grid-item"
                            :style="{ backgroundColor: layerCellColor(item) }" />
                    </tr>
                </tbody>
            </table>
        </div>

        <h6 class="section-title extras-title">Extras</h6>
        <div class="extras-grid">
            <div v-for="item in extrasEntries" :key="item.index" class="extra-card">
                <div class="extra-label">{{ item.label }}</div>
                <div class="extra-value">{{ item.value.toFixed(3) }}</div>
                <div class="extra-bar-track">
                    <span class="extra-bar-fill" :style="{ width: normalizedExtraWidth(item.value) }" />
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
    obs: number[][][],
    extras: number[],
    extrasMeanings: string[]
}>();

const extraMin = computed(() => {
    if (props.extras.length === 0) return 0;
    return Math.min(...props.extras);
});

const extraMax = computed(() => {
    if (props.extras.length === 0) return 1;
    const maxValue = Math.max(...props.extras);
    return maxValue === extraMin.value ? maxValue + 1 : maxValue;
});

const extrasEntries = computed(() => {
    return props.extras.map((value, index) => ({
        index,
        value,
        label: props.extrasMeanings[index] ?? `extra_${index + 1}`,
    }));
});

function layerCellColor(value: number) {
    if (value == 1) return "red";
    if (value == -1) return "blue";
    if (value == 0) return "white";
    return "black";
}

function normalizedExtraWidth(value: number): string {
    const ratio = (value - extraMin.value) / (extraMax.value - extraMin.value);
    const clipped = Math.max(0, Math.min(1, ratio));
    return `${Math.max(8, clipped * 100)}%`;
}



</script>
<style scoped>
.observation-preview {
    display: grid;
    gap: 0.55rem;
}

.section-title {
    margin: 0;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--bs-secondary-color);
}

.extras-title {
    margin-top: 0.2rem;
}

.layers-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}

.layer-table {
    margin: 0;
}

.grid-item {
    width: 10px;
    height: 10px;
    border: 1px solid color-mix(in srgb, var(--bs-border-color) 65%, transparent);
}

.extras-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.4rem;
}

.extra-card {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.45rem;
    background: color-mix(in srgb, var(--bs-secondary-bg) 75%, var(--bs-body-bg));
    padding: 0.35rem 0.45rem;
    display: grid;
    gap: 0.2rem;
    text-align: left;
}

.extra-label {
    font-size: 0.72rem;
    color: var(--bs-secondary-color);
    overflow-wrap: anywhere;
}

.extra-value {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    font-size: 0.8rem;
}

.extra-bar-track {
    height: 0.28rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--bs-border-color) 40%, transparent);
    overflow: hidden;
}

.extra-bar-fill {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg,
            color-mix(in srgb, var(--bs-info) 65%, #fff),
            color-mix(in srgb, var(--bs-primary) 72%, #fff));
}
</style>