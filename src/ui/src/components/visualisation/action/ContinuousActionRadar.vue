<template>
    <div class="continuous-radar">
        <p v-if="dimensionCount === 0" class="text-muted mb-0">No continuous action vector available at this step.</p>

        <template v-else>
            <svg class="radar-svg" viewBox="0 0 240 240" role="img" aria-label="Continuous action radar chart">
                <g>
                    <polygon v-for="ring in rings" :key="ring" :points="ringPoints(ring)" class="grid-ring" />
                    <line v-for="axis in axes" :key="`axis-${axis.index}`" :x1="center" :y1="center" :x2="axis.x"
                        :y2="axis.y" class="axis-line" />

                    <polygon :points="currentPolygonPoints" class="current-polygon" />

                    <text v-for="axis in axes" :key="`label-${axis.index}`" :x="axis.labelX" :y="axis.labelY"
                        class="axis-label">
                        A{{ axis.index + 1 }}
                    </text>
                </g>
            </svg>

            <div class="radar-values">
                <div v-for="dimension in dimensions" :key="dimension.index" class="dimension-row">
                    <span>D{{ dimension.index + 1 }}</span>
                    <span>{{ formatValue(dimension.current) }}</span>
                </div>
            </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { ContinuousActionSpace } from '../../../models/Env';
import { ActionValue, ReplayEpisode } from '../../../models/Episode';

const props = defineProps<{
    episode: ReplayEpisode
    currentStep: number
    selectedAgents: number[]
    actionSpace: ContinuousActionSpace
}>();

const center = 120;
const radius = 82;
const rings = [0.25, 0.5, 0.75, 1];

const safeStep = computed(() => clampStep(props.currentStep));

const currentVector = computed(() => aggregateVectorAt(safeStep.value));

const dimensionCount = computed(() => {
    const fromCurrent = currentVector.value.length;
    const fromShape = props.actionSpace.shape.at(-1) ?? 0;
    return Math.max(fromCurrent, fromShape);
});

const dimensions = computed(() => {
    return Array.from({ length: dimensionCount.value }, (_, index) => ({
        index,
        current: currentVector.value[index] ?? 0,
    }));
});

const axes = computed(() => {
    return Array.from({ length: dimensionCount.value }, (_, index) => {
        const angle = angleForDimension(index, dimensionCount.value);
        const x = center + radius * Math.cos(angle);
        const y = center + radius * Math.sin(angle);
        const labelX = center + (radius + 14) * Math.cos(angle);
        const labelY = center + (radius + 14) * Math.sin(angle);
        return { index, x, y, labelX, labelY };
    });
});

const currentPolygonPoints = computed(() => polygonPoints(currentVector.value));

function clampStep(step: number): number {
    const max = Math.max(0, props.episode.episode.actions.length - 1);
    return Math.max(0, Math.min(max, step));
}

function aggregateVectorAt(step: number): number[] {
    const vectors = props.selectedAgents
        .map((agent) => toVector(props.episode.episode.actions[step]?.[agent]))
        .filter((vector): vector is number[] => vector.length > 0);

    if (vectors.length === 0) return [];

    const dims = Math.max(...vectors.map((vector) => vector.length));
    return Array.from({ length: dims }, (_, dim) => {
        const values = vectors.map((vector) => vector[dim] ?? 0);
        return values.reduce((sum, value) => sum + value, 0) / values.length;
    });
}

function toVector(value: ActionValue | undefined): number[] {
    if (Array.isArray(value)) {
        return value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item));
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
        return [value];
    }
    return [];
}

function angleForDimension(index: number, total: number): number {
    if (total <= 0) return -Math.PI / 2;
    return -Math.PI / 2 + (index / total) * Math.PI * 2;
}

function normalize(value: number, index: number): number {
    const low = props.actionSpace.low?.[index];
    const high = props.actionSpace.high?.[index];

    if (low != null && high != null && high > low) {
        return Math.max(0, Math.min(1, (value - low) / (high - low)));
    }

    const fallbackMin = -1;
    const fallbackMax = 1;
    return Math.max(0, Math.min(1, (value - fallbackMin) / (fallbackMax - fallbackMin)));
}

function polygonPoints(vector: number[]): string {
    if (dimensionCount.value === 0) return '';

    const points = Array.from({ length: dimensionCount.value }, (_, index) => {
        const angle = angleForDimension(index, dimensionCount.value);
        const value = vector[index] ?? 0;
        const length = normalize(value, index) * radius;
        const x = center + length * Math.cos(angle);
        const y = center + length * Math.sin(angle);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    return points.join(' ');
}

function ringPoints(ringRatio: number): string {
    if (dimensionCount.value === 0) return '';
    const points = Array.from({ length: dimensionCount.value }, (_, index) => {
        const angle = angleForDimension(index, dimensionCount.value);
        const x = center + radius * ringRatio * Math.cos(angle);
        const y = center + radius * ringRatio * Math.sin(angle);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    return points.join(' ');
}

function formatValue(value: number): string {
    return Number.isFinite(value) ? value.toFixed(3) : '-';
}

</script>

<style scoped>
.radar-svg {
    width: 100%;
    max-height: 17rem;
    display: block;
    background: color-mix(in srgb, var(--bs-body-bg) 80%, transparent);
    border-radius: 0.5rem;
    border: 1px solid var(--bs-border-color);
}

.grid-ring {
    fill: none;
    stroke: color-mix(in srgb, var(--bs-secondary-color) 28%, transparent);
    stroke-width: 1;
}

.axis-line {
    stroke: color-mix(in srgb, var(--bs-secondary-color) 38%, transparent);
    stroke-width: 1;
}

.current-polygon {
    fill: color-mix(in srgb, var(--bs-primary) 30%, transparent);
    stroke: color-mix(in srgb, var(--bs-primary) 65%, #000);
    stroke-width: 2;
}

.axis-label {
    font-size: 10px;
    fill: var(--bs-secondary-color);
    text-anchor: middle;
    dominant-baseline: central;
}

.radar-values {
    display: grid;
    gap: 0.2rem;
}

.dimension-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
    font-size: 0.75rem;
}

.dimension-row span:nth-child(2) {
    font-variant-numeric: tabular-nums;
    justify-self: end;
}
</style>
