<template>
    <ul class="nav nav-tabs">
        <li v-for="tabName in tabs" class="nav-item" @click="() => changeTab(tabName)">
            <a href="#" class="nav-link" :class="currentTab == tabName ? 'active' : ''"> {{ tabName }} </a>
        </li>
    </ul>
</template>

<script setup lang="ts">
import { ref } from 'vue';


const tabs = [
    "Overview",
    "Inspect"
];
const currentTab = ref(tabs[0]);

const emit = defineEmits(["tabChange"]);

function changeTab(newTab: string) {
    currentTab.value = newTab;
    emit("tabChange", newTab);
}


defineExpose({ changeTab })

</script>