<template>
    <main>
        <header class="app-header">
            <h1 class="app-title">
                <RouterLink to="/home">
                    <img src="./assets/logo.svg" alt="Marl Logo" class="me-2" style="height: 1em; vertical-align: middle" />
                    MARL-UI
                </RouterLink>
            </h1>
            <div class="app-header-actions">
                <button type="button" class="btn btn-sm btn-outline-secondary" @click="openSettings">
                    <font-awesome-icon :icon="['fas', 'gear']" class="me-1" />
                    Settings
                </button>
                <SystemInfo />
            </div>
        </header>
        <SettingsModal ref="settingsModal" />
        <RouterView></RouterView>
        <GlobalErrors />
    </main>
</template>

<script setup lang="ts">
import { ref } from "vue";
import SystemInfo from "./components/SystemInfo.vue";
import GlobalErrors from "./components/GlobalErrors.vue";
import SettingsModal from "./components/modals/SettingsModal.vue";

const settingsModal = ref<InstanceType<typeof SettingsModal> | null>(null);

function openSettings() {
    settingsModal.value?.showModal();
}
</script>

<style>
:root {
    font-synthesis: none;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    -webkit-text-size-adjust: 100%;
}

html,
body,
#app,
main {
    height: 100%;
    max-height: 100%;
    background-color: var(--bs-body-bg);
    color: var(--bs-body-color);
}

main {
    overflow-x: hidden;
    box-sizing: border-box;
    padding-left: 1%;
    padding-right: 1%;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.35rem;
}

.app-title {
    margin: 0;
}

.app-title a {
    color: inherit;
    text-decoration: none;
}

.app-title a:hover,
.app-title a:focus-visible {
    text-decoration: underline;
}

.app-header-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

dialog {
    top: 50%;
    left: 50%;
    translate: -50% -50%;
    border-radius: 3%;
}
</style>
