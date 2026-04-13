<template>
    <Card v-if="isRoot" class="json-inspector-card">
        <template #title>
            <span class="json-inspector-title">{{ title }}</span>
        </template>

        <template #content>
            <div class="json-inspector-body">
                <JsonInspector v-for="entry in entries" :key="entry.key" :label="entry.key" :value="entry.value"
                    :depth="0" />
                <p v-if="entries.length === 0" class="text-muted mb-0">No parameters available</p>
            </div>
        </template>
    </Card>

    <div v-else class="json-node" :style="nodeStyle">
        <div class="json-row">

            <span class="json-key">
                <span class="json-toggle-slot">
                    <Button v-if="expandable" class="json-toggle" :label="expanded ? '-' : '+'" severity="secondary"
                        text rounded size="small" @click="expanded = !expanded" />
                </span>
                {{ formatLabel(label) }}: &nbsp;
            </span>
            <div class="json-value">
                <span v-if="!expandable" class="json-primitive">{{ formatPrimitive(value) }}</span>
                <span v-else-if="nameValue !== null" class="json-name">{{ nameValue }}</span>
            </div>
        </div>

        <div v-if="expandable && expanded" class="json-children">
            <JsonInspector v-for="entry in entries" :key="entry.key" :label="entry.key" :value="entry.value"
                :depth="depth + 1" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import Button from 'primevue/button';
import Card from 'primevue/card';

defineOptions({ name: 'JsonInspector' });

type JsonEntry = {
    key: string;
    value: unknown;
};

const props = withDefaults(defineProps<{
    title?: string;
    label?: string;
    value: unknown;
    depth?: number;
}>(), {
    depth: 0,
});

const expanded = ref(false);

const isRoot = computed(() => props.label == null);
const entries = computed(() => extractEntries(props.value));
const expandable = computed(() => entries.value.length > 0);
const nodeStyle = computed(() => ({
    '--json-indent': `${props.depth * 0.9}rem`,
}));
const nameValue = computed(() => {
    if (!expandable.value) return null;
    if (typeof props.value !== 'object' || props.value == null) return null;
    if (Array.isArray(props.value) || props.value instanceof Map || props.value instanceof Set) return null;
    const obj = props.value as Record<string, unknown>;
    if ('name' in obj && typeof obj.name === 'string') {
        return obj.name;
    }
    return null;
});

function extractEntries(value: unknown): JsonEntry[] {
    if (value == null) return [];

    if (Array.isArray(value)) {
        return value.map((item, index) => ({ key: `[${index}]`, value: item }));
    }

    if (value instanceof Map) {
        return Array.from(value.entries()).map(([key, entryValue]) => ({
            key: formatKey(key),
            value: entryValue,
        }));
    }

    if (value instanceof Set) {
        return Array.from(value.values()).map((entryValue, index) => ({
            key: `[${index}]`,
            value: entryValue,
        }));
    }

    if (typeof value === 'object') {
        return Object.entries(value as Record<string, unknown>).map(([key, entryValue]) => ({ key, value: entryValue }));
    }

    return [];
}

function formatPrimitive(value: unknown): string {
    if (value == null) return 'null';
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
        return String(value);
    }
    if (value instanceof Date) return value.toISOString();
    if (Array.isArray(value)) return `Array(${value.length})`;
    if (value instanceof Map) return `Map(${value.size})`;
    if (value instanceof Set) return `Set(${value.size})`;
    if (typeof value === 'object') return `${value.constructor?.name ?? 'Object'}`;
    return String(value);
}

function formatKey(key: unknown): string {
    if (typeof key === 'string') return key;
    if (typeof key === 'number' || typeof key === 'boolean' || typeof key === 'bigint') return String(key);
    if (key instanceof Date) return key.toISOString();
    return String(key);
}

function formatLabel(label: string | undefined): string {
    if (!label) return '';
    // Don't format array indices or other special keys
    if (label.startsWith('[') && label.endsWith(']')) {
        return label;
    }
    // Replace underscores with spaces and capitalize first word
    const words = label.split('_');
    const formatted = words.map((word, index) => {
        if (!word) return '';
        if (index === 0) {
            // First word: capitalize first letter, rest lowercase
            return word[0].toUpperCase() + word.slice(1).toLowerCase();
        }
        // Other words: keep lowercase
        return word.toLowerCase();
    }).join(' ');
    return formatted;
}
</script>

<style scoped>
.json-inspector-card {
    width: 100%;
}

.json-inspector-title {
    font-weight: 700;
}

.json-inspector-body {
    display: grid;
    gap: 0.2rem;
}

.json-node {
    padding-left: var(--json-indent);
}

.json-row {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.5rem;
    align-items: center;
}

.json-key {
    display: inline-flex;
    align-items: center;
    gap: 0.15rem;
    font-weight: 600;
    color: var(--bs-body-color);
    line-height: 1.2;
    white-space: nowrap;
}

.json-value {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    min-width: 0;
    flex-wrap: wrap;
}

.json-primitive {
    color: var(--bs-secondary-color);
    overflow-wrap: anywhere;
    line-height: 1.2;
}

.json-name {
    color: var(--bs-body-color);
    overflow-wrap: anywhere;
    line-height: 1.2;
}

.json-children {
    margin-top: 0.2rem;
    padding-left: 0.65rem;
    border-left: 1px solid var(--bs-border-color);
    display: grid;
    gap: 0.2rem;
}

.json-toggle {
    width: 1.05rem;
    height: 1.05rem;
    padding: 0;
    flex-shrink: 0;
}

.json-toggle-slot {
    width: 1.1rem;
    display: inline-flex;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
}
</style>