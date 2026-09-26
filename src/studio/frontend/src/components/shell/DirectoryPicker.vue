<script setup lang="ts">
/** Browse directories on the Studio server; browser-native directory pickers cannot expose server paths. */
import { nextTick, onMounted, ref } from "vue";
import { useApi, type DirectoryListing } from "../../api";

const props = defineProps<{ initial?: string }>();
const emit = defineEmits<{ select: [path: string]; close: [] }>();
const listing = ref<DirectoryListing | null>(null);
const dialog = ref<HTMLElement | null>(null);
const loading = ref(false);
const error = ref("");
let requestId = 0;

/** Only the latest navigation response may replace the visible directory. @ai-generated */
async function browse(path?: string): Promise<void> {
    const current = ++requestId;
    loading.value = true;
    error.value = "";
    try {
        const result = await useApi().browseDirectories(path);
        if (current === requestId) listing.value = result;
    } catch (e) {
        if (current === requestId) {
            if (path && !listing.value) void browse();
            else error.value = (e as Error).message;
        }
    } finally {
        if (current === requestId) loading.value = false;
    }
}

onMounted(() => {
    void browse(props.initial);
    void nextTick(() => dialog.value?.focus());
});
</script>

<template>
    <Teleport to="body">
        <div class="picker-scrim" @click.self="emit('close')" @keydown.esc.stop="emit('close')">
            <section ref="dialog" class="picker" role="dialog" aria-modal="true" aria-label="Choose log directory" tabindex="-1">
                <header>
                    <h2>Choose log directory</h2>
                    <button type="button" class="btn" aria-label="Close directory browser" @click="emit('close')">Close</button>
                </header>
                <p class="muted">Folders on the Studio server</p>
                <p v-if="listing" class="current" :title="listing.path">{{ listing.path }}</p>
                <p v-if="error" role="alert" class="error">{{ error }}</p>
                <div class="folders" aria-label="Directories">
                    <button v-if="listing?.parent" type="button" :disabled="loading" @click="browse(listing.parent)">.. (up)</button>
                    <button
                        v-for="dir in listing?.directories ?? []"
                        :key="dir.path"
                        type="button"
                        :disabled="loading"
                        @click="browse(dir.path)"
                    >
                        {{ dir.name }}/
                    </button>
                    <p v-if="loading" role="status">Loading directories…</p>
                    <p v-else-if="listing && !listing.directories.length" class="muted">No subdirectories</p>
                </div>
                <footer>
                    <button type="button" class="btn" @click="emit('close')">Cancel</button>
                    <button
                        type="button"
                        class="btn primary"
                        :disabled="!listing || loading"
                        @click="listing && emit('select', listing.path)"
                    >
                        Select this directory
                    </button>
                </footer>
            </section>
        </div>
    </Teleport>
</template>

<style scoped>
.picker-scrim {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: grid;
    place-items: center;
    background: rgb(0 0 0 / 50%);
}
.picker {
    width: min(560px, calc(100vw - 32px));
    max-height: min(680px, calc(100vh - 32px));
    display: flex;
    flex-direction: column;
    padding: 24px;
    border-radius: 12px;
    background: var(--card);
    color: var(--ink);
}
header,
footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
h2 {
    margin: 0;
    font-size: 18px;
}
.current {
    overflow-wrap: anywhere;
    font-family: monospace;
}
.error {
    color: var(--err);
}
.folders {
    overflow-y: auto;
    min-height: 120px;
    margin: 12px 0 20px;
    border: 1px solid var(--line);
    border-radius: 6px;
}
.folders button {
    display: block;
    width: 100%;
    padding: 10px 12px;
    border: 0;
    background: transparent;
    color: var(--ink);
    text-align: left;
    cursor: pointer;
}
.folders button:hover,
.folders button:focus-visible {
    background: var(--acc-soft);
}
.folders p {
    padding: 0 12px;
}
</style>
