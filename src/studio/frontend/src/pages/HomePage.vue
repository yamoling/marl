<script setup lang="ts">
/** Workspace picker; the server's selected workspace is only a suggestion. */
import { computed, nextTick, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { useNamedWorkspacesStore } from "../stores/namedWorkspaces";
import { useWorkspaceStore } from "../stores/workspace";
import { readStorage } from "../stores/storage";
import { restoreWorkspace, STORAGE_KEY } from "../domain/workspace";
import { displayNames } from "../domain/format";
import Icon from "../components/shell/Icon.vue";
import DirectoryPicker from "../components/shell/DirectoryPicker.vue";

const router = useRouter();
const workspaces = useNamedWorkspacesStore();
const plotting = useWorkspaceStore();

/** Preview each workspace's persisted loaded experiments, including edits not yet flushed to storage. @ai-generated */
const previews = computed(() =>
    Object.fromEntries(
        workspaces.workspaces.map((w) => {
            const key = `${STORAGE_KEY}.${encodeURIComponent(w.id)}`;
            const raw = readStorage(key) ?? (w.id === workspaces.selected ? readStorage(STORAGE_KEY) : null);
            const state = w.id === workspaces.activeId ? plotting.ws : raw === null ? null : restoreWorkspace(raw).workspace;
            const ids = state?.experiments ?? [];
            return [w.id, { ids, names: displayNames(ids), colours: state?.colours ?? {} }];
        }),
    ),
);

const name = ref("");
const newLogdir = ref("");
const creating = ref(false);
const editingId = ref<string | null>(null);
const editedName = ref("");
const editingRootId = ref<string | null>(null);
const editedRoot = ref("");
const pickerFor = ref<"new" | "root" | null>(null);
const pickerInitial = computed(() =>
    pickerFor.value === "root"
        ? editedRoot.value
        : newLogdir.value || workspaces.workspaces.find((w) => w.id === workspaces.selected)?.logdir,
);
const busy = ref(false);
const error = ref("");

/** Keep failed mutations visible and allow the user to retry without losing form input. @ai-generated */
async function perform(action: () => Promise<void>): Promise<void> {
    busy.value = true;
    error.value = "";
    try {
        await action();
    } catch (e) {
        error.value = (e as Error).message;
    } finally {
        busy.value = false;
    }
}

/** Create a workspace and leave it on the home screen for explicit entry. @ai-generated */
function create(): void {
    if (!name.value.trim()) return;
    void perform(async () => {
        await workspaces.create(name.value, newLogdir.value.trim() || undefined);
        name.value = "";
        newLogdir.value = "";
        creating.value = false;
    });
}

/** Open the create form and focus its name field. @ai-generated */
function startCreating(): void {
    creating.value = true;
    void nextTick(() => document.getElementById("new-name")?.focus());
}

/** Make the card clickable without triggering navigation from its embedded controls. @ai-generated */
function openCard(event: MouseEvent, id: string): void {
    if (!busy.value && !(event.target as Element).closest("button, input, form, label")) enter(id);
}

/** Enter the workspace only after the server has confirmed its selection. @ai-generated */
function enter(id: string): void {
    void perform(async () => {
        await workspaces.enter(id);
        await router.push({ name: "studio" });
    });
}

/** Edit the title in place and focus it for immediate typing. @ai-generated */
function editName(id: string, name: string): void {
    editedName.value = name;
    editingId.value = id;
    void nextTick(() => document.getElementById(`name-${id}`)?.focus());
}

/** Enter commits the inline title; a failed request leaves the editor open. @ai-generated */
async function rename(id: string): Promise<void> {
    if (busy.value || !editedName.value.trim()) return;
    busy.value = true;
    error.value = "";
    try {
        await workspaces.rename(id, editedName.value);
        editingId.value = null;
    } catch (e) {
        error.value = (e as Error).message;
    } finally {
        busy.value = false;
    }
}

/** Save an edited root without changing the workspace name or its plotting state. @ai-generated */
function editRoot(id: string, logdir: string): void {
    editingRootId.value = id;
    editedRoot.value = logdir;
}

/** Keep the root editor open if the directory is rejected by the server. @ai-generated */
async function saveRoot(id: string): Promise<void> {
    if (busy.value || !editedRoot.value.trim()) return;
    await perform(async () => {
        await workspaces.setLogdir(id, editedRoot.value);
        editingRootId.value = null;
    });
}

/** Fill the relevant editable field with the directory selected on the server. @ai-generated */
function chooseDirectory(path: string): void {
    if (pickerFor.value === "root") editedRoot.value = path;
    else newLogdir.value = path;
    pickerFor.value = null;
}

/** Delete workspace configuration after confirmation, without deleting any files. @ai-generated */
function trash(id: string, name: string): void {
    if (!window.confirm(`Trash workspace “${name}”? Experiments on disk will not be deleted.`)) return;
    void perform(() => workspaces.trash(id));
}

onMounted(() => void perform(() => workspaces.refresh()));
</script>

<template>
    <main class="home">
        <header>
            <h1>MARL Studio</h1>
            <p>Choose a workspace to explore its experiments and plots.</p>
        </header>
        <p v-if="error" role="alert" class="error">{{ error }}</p>
        <section aria-label="Workspaces" class="cards">
            <article class="card create-card">
                <button v-if="!creating" type="button" class="create-trigger" :disabled="busy" @click="startCreating">
                    <span class="create-icon"><Icon name="plus" :size="36" :stroke-width="1.5" /></span>
                    <span>Add a new workspace</span>
                </button>
                <form v-else class="create-form" @submit.prevent="create" @keydown.esc.prevent="creating = false">
                    <h2>New workspace</h2>
                    <label for="new-name">Workspace name</label>
                    <input id="new-name" v-model="name" placeholder="My workspace" :disabled="busy" />
                    <label for="new-logdir">Root log directory (optional; defaults to selected workspace)</label>
                    <div class="path-entry">
                        <input id="new-logdir" v-model="newLogdir" placeholder="/path/to/logs" :disabled="busy" />
                        <button type="button" class="btn" :disabled="busy" @click="pickerFor = 'new'">Browse…</button>
                    </div>
                    <div class="row">
                        <button type="button" class="btn" :disabled="busy" @click="creating = false">Cancel</button
                        ><button class="btn primary" :disabled="busy || !name.trim()" type="submit">Create</button>
                    </div>
                </form>
            </article>
            <article
                v-for="w in workspaces.workspaces"
                :key="w.id"
                class="card workspace-card"
                role="button"
                tabindex="0"
                :aria-label="`Open ${w.name}`"
                @click="openCard($event, w.id)"
                @keydown.enter.self.prevent="enter(w.id)"
                @keydown.space.self.prevent="enter(w.id)"
            >
                <div class="heading">
                    <h2>
                        <input
                            v-if="editingId === w.id"
                            :id="`name-${w.id}`"
                            v-model="editedName"
                            :aria-label="`Rename ${w.name}`"
                            :disabled="busy"
                            @keydown.enter.prevent="rename(w.id)"
                            @keydown.esc.prevent="editingId = null"
                        />
                        <button
                            v-else
                            type="button"
                            class="title"
                            :aria-label="`Rename ${w.name}`"
                            :disabled="busy"
                            @click="editName(w.id, w.name)"
                        >
                            {{ w.name }}
                        </button>
                    </h2>
                    <span v-if="w.id === workspaces.selected" class="muted">Selected</span>
                    <button
                        type="button"
                        class="trash"
                        :disabled="busy"
                        :aria-label="`Trash ${w.name}`"
                        :title="`Trash ${w.name}`"
                        @click="trash(w.id, w.name)"
                    >
                        <Icon name="trash" :size="17" />
                    </button>
                </div>
                <form
                    v-if="editingRootId === w.id"
                    class="root-form"
                    @submit.prevent="saveRoot(w.id)"
                    @keydown.esc.prevent="editingRootId = null"
                >
                    <label :for="`root-${w.id}`">Root log directory</label>
                    <div class="path-entry">
                        <input :id="`root-${w.id}`" v-model="editedRoot" :disabled="busy" />
                        <button type="button" class="btn" :disabled="busy" @click="pickerFor = 'root'">Browse…</button>
                    </div>
                    <button class="btn" type="submit" :disabled="busy || !editedRoot.trim()">Save</button>
                    <button class="btn" type="button" @click="editingRootId = null">Cancel</button>
                </form>
                <button v-else type="button" class="root-path muted" :title="w.logdir" :disabled="busy" @click="editRoot(w.id, w.logdir)">
                    {{ w.logdir }}
                </button>
                <div class="experiments-preview">
                    <strong
                        >{{ previews[w.id]?.ids.length ?? 0 }} loaded
                        {{ previews[w.id]?.ids.length === 1 ? "experiment" : "experiments" }}</strong
                    >
                    <div v-if="previews[w.id]?.ids.length" class="preview-pills" :aria-label="`Loaded experiments in ${w.name}`">
                        <span v-for="id in previews[w.id].ids" :key="id" class="xpill preview-pill" :title="id">
                            <span class="dot" :style="{ '--c': previews[w.id].colours[id] ?? '#b9b9c4' }" />
                            <span class="preview-name">{{ previews[w.id].names[id] }}</span>
                        </span>
                    </div>
                </div>
            </article>
        </section>
        <DirectoryPicker v-if="pickerFor" :initial="pickerInitial" @select="chooseDirectory" @close="pickerFor = null" />
    </main>
</template>

<style scoped>
.home {
    max-width: 1040px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    overflow: auto;
    height: 100%;
}
.home header {
    margin-bottom: 28px;
}
h1 {
    font-size: 30px;
    margin: 0 0 8px;
}
h2 {
    font-size: 18px;
    margin: 0;
}
p {
    color: var(--ink2);
}
.error {
    color: var(--danger, #b42318);
}
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
    gap: 16px;
}
.card {
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--card);
    padding: 20px;
    box-shadow: var(--sh);
}
.workspace-card {
    cursor: pointer;
    transition:
        border-color 150ms,
        box-shadow 150ms;
}
.workspace-card:hover,
.workspace-card:focus-visible {
    border-color: var(--acc);
    box-shadow: var(--sh-hi);
}
.create-card {
    min-height: 238px;
    padding: 0;
}
.create-trigger {
    width: 100%;
    height: 100%;
    min-height: 238px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    border: 0;
    border-radius: var(--r);
    background: transparent;
    color: var(--ink2);
    font: inherit;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
}
.create-trigger:hover,
.create-trigger:focus-visible {
    background: var(--acc-soft);
    color: var(--acc2);
}
.create-icon {
    display: grid;
    place-items: center;
    width: 72px;
    height: 72px;
    border: 1px dashed var(--line-hover);
    border-radius: 20px;
    color: var(--acc);
    background: var(--acc-soft);
}
.create-trigger:hover .create-icon,
.create-trigger:focus-visible .create-icon {
    border-color: var(--acc);
}
.create-form {
    padding: 20px;
}
.create-form h2 {
    margin-bottom: 28px;
}
.create-form input {
    width: 100%;
    margin-bottom: 14px;
}
.create-form .row {
    justify-content: flex-end;
}
.heading,
.row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.heading {
    margin-bottom: 20px;
}
.heading h2 {
    flex: 1;
    min-width: 0;
}
.title {
    font: inherit;
    font-weight: inherit;
    color: inherit;
    background: none;
    border: 0;
    padding: 0;
    cursor: text;
    text-align: left;
    overflow-wrap: anywhere;
}
.title:hover {
    text-decoration: underline;
}
.heading input {
    width: 100%;
    font: inherit;
}
.trash {
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    margin-left: auto;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--ink3);
    cursor: pointer;
}
.trash:hover,
.trash:focus-visible {
    color: var(--err);
    background: var(--err-soft);
}
.trash:disabled {
    opacity: 0.5;
    cursor: default;
}
label,
strong {
    display: block;
    margin-bottom: 8px;
    font-size: 13px;
}
input {
    min-width: 0;
    flex: 1;
    padding: 8px;
    border: 1px solid var(--line2);
    border-radius: 6px;
    background: var(--card);
    color: var(--ink);
}
.root-path {
    display: block;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: left;
    border: 0;
    background: none;
    cursor: text;
}
.path-entry {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.path-entry input {
    width: 100%;
    margin-bottom: 0;
}
.path-entry button {
    flex: none;
}
.experiments-preview {
    margin: 20px 0;
}
.preview-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.preview-pill {
    padding-right: 10px;
    max-width: 100%;
}
.preview-name {
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
