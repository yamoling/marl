<template>
    <ul class="nav nav-tabs mb-2">
        <li v-for="tabName in tabs" class="nav-item" @click="() => changeTab(tabName)">
            <a href="#" class="nav-link" :class="currentTab == tabName ? 'active' : ''"> {{ tabName }} </a>
        </li>
    </ul>
</template>

<script setup lang="ts">
import { ref } from 'vue';
export interface ITabs<T extends readonly string[]> {
    changeTab: (newTab: T[number]) => void,
    currentTab: T[number]
};

const props = defineProps<{
    tabs: readonly string[]
}>();
const currentTab = ref(props.tabs[0]);

const emit = defineEmits(["tabChange"]);

function changeTab(newTab: string) {
    currentTab.value = newTab;
    emit("tabChange", newTab);
}


defineExpose({ changeTab, currentTab })
</script>