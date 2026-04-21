<template>
    <Teleport to="body">
        <div class="toast-container">
            <TransitionGroup name="toast">
                <div
                    v-for="error in errorStore.errors"
                    :key="error.id"
                    class="card border-danger toast-card"
                >
                    <!-- Header row -->
                    <div
                        class="card-header d-flex align-items-center gap-2 py-2"
                    >
                        <font-awesome-icon
                            icon="exclamation-circle"
                            class="text-danger flex-shrink-0"
                        />
                        <span class="fw-bold flex-grow-1">{{
                            error.title
                        }}</span>
                        <button
                            type="button"
                            class="btn-close"
                            aria-label="Dismiss"
                            @click="errorStore.dismiss(error.id)"
                        />
                    </div>

                    <!-- Error type badge (Python exception class name) -->
                    <div v-if="error.errorType" class="px-3 pt-2">
                        <code class="error-type-badge">{{
                            error.errorType
                        }}</code>
                    </div>

                    <!-- Detail block -->
                    <div class="card-body py-2">
                        <pre class="error-detail">{{ error.detail }}</pre>
                    </div>

                    <!-- Footer: relative timestamp -->
                    <div class="card-footer text-muted py-1">
                        <small>{{ formatTimestamp(error.timestamp) }}</small>
                    </div>
                </div>
            </TransitionGroup>
        </div>
    </Teleport>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
import { useErrorStore } from "../stores/ErrorStore";

const errorStore = useErrorStore();

// ── Reactive clock so timestamps re-render every second ────────────────────
const now = ref(Date.now());
const clockInterval = setInterval(() => {
    now.value = Date.now();
}, 1_000);

onUnmounted(() => {
    clearInterval(clockInterval);
});

function formatTimestamp(date: Date): string {
    const elapsed = (now.value - date.getTime()) / 1_000;
    if (elapsed < 5) return "just now";
    if (elapsed < 60) return `${Math.floor(elapsed)}s ago`;
    return date.toLocaleTimeString();
}

// ── Auto-dismiss logic ─────────────────────────────────────────────────────
const timers = new Map<number, ReturnType<typeof setTimeout>>();

watch(
    () => errorStore.errors,
    (currentErrors) => {
        // Schedule a timer for every new autoDismiss error we haven't seen yet
        for (const error of currentErrors) {
            if (error.autoDismiss && !timers.has(error.id)) {
                const timer = setTimeout(() => {
                    errorStore.dismiss(error.id);
                    timers.delete(error.id);
                }, 10_000);
                timers.set(error.id, timer);
            }
        }

        // Cancel and remove timers whose errors have already been dismissed
        const activeIds = new Set(currentErrors.map((e) => e.id));
        for (const [id, timer] of timers) {
            if (!activeIds.has(id)) {
                clearTimeout(timer);
                timers.delete(id);
            }
        }
    },
    { deep: true },
);
</script>

<style scoped>
/* ── Container ────────────────────────────────────────────────────────────── */
.toast-container {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    z-index: 9999;
    max-width: 420px;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    /* Don't block pointer events in empty space */
    pointer-events: none;
}

.toast-card {
    pointer-events: all;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

/* ── Error-type badge ─────────────────────────────────────────────────────── */
.error-type-badge {
    display: inline-block;
    font-size: 0.75rem;
    background-color: #f1f3f5;
    color: #6c757d;
    border-radius: 0.25rem;
    padding: 0.1em 0.5em;
}

/* ── Detail pre-block ─────────────────────────────────────────────────────── */
.error-detail {
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 10rem;
    overflow-y: auto;
    font-size: 0.78rem;
    margin: 0;
}

/* ── TransitionGroup: slide in from the right, fade out to the right ──────── */
.toast-enter-active {
    transition:
        opacity 0.25s ease,
        transform 0.25s ease;
}

.toast-leave-active {
    transition:
        opacity 0.3s ease,
        transform 0.3s ease;
    /* Take out of flow so remaining toasts animate up smoothly */
    position: absolute;
    width: 100%;
}

.toast-move {
    transition: transform 0.3s ease;
}

.toast-enter-from {
    opacity: 0;
    transform: translateX(3rem);
}

.toast-leave-to {
    opacity: 0;
    transform: translateX(3rem);
}
</style>
