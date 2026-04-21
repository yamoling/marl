<template>
    <main>
        <header class="app-header">
            <h1 class="app-title">
                <RouterLink to="/home" aria-label="Go to homepage"
                    >RL Dashboard</RouterLink
                >
            </h1>
            <SystemInfo />
        </header>
        <RouterView></RouterView>
        <GlobalErrors />
    </main>
</template>

<script setup lang="ts">
import SystemInfo from "./components/SystemInfo.vue";
import GlobalErrors from "./components/GlobalErrors.vue";
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

dialog {
    top: 50%;
    left: 50%;
    translate: -50% -50%;
    border-radius: 3%;
}
</style>
