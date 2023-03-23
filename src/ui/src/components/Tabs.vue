<template>
    <ul class="nav nav-tabs mb-2">
        <li @click="() => changeTab('Home')">
            <a href="#" class="nav-link" :class="currentTab == 'Home' ? 'active' : ''">
                Home
            </a>
        </li>
        <li v-for="(tabName, index) in tabs" class="nav-item" @click="() => changeTab(tabName)">
            <a v-if="index > 0" href="#" class="nav-link" style="padding-right: 35px;"
                :class="currentTab == tabName ? 'active' : ''">
                {{ tabName }}
                <button v-if="tabName == currentTab" style="position: absolute;" class="btn btn-sm"
                    @click.stop="() => deleteTab(tabName)">
                    <font-awesome-icon :icon="['far', 'circle-xmark']" />
                </button>
            </a>
        </li>
    </ul>
</template>

<script setup lang="ts">
import { ref } from 'vue';
export interface ITabs {
    addTab(tabName: string): void
    changeTab: (newTab: string) => void
    deleteTab: (tabName: string) => void
    currentTab: string
};


const tabs = ref(["Home"]);
const currentTab = ref(tabs.value[0]);
const emit = defineEmits(["tabChange", "tabDelete"]);

function changeTab(newTab: string) {
    currentTab.value = newTab;
    emit("tabChange", newTab);
}

function deleteTab(tabName: string) {
    const tabIndex = tabs.value.indexOf(tabName);
    if (tabIndex > 0) {
        tabs.value.splice(tabIndex, 1);
        emit("tabDelete", tabName);
        // If the deleted tab is the last one, change to the previous tab
        if (tabIndex == tabs.value.length) {
            changeTab(tabs.value[tabIndex - 1]);
        }
    }
}

function addTab(tabName: string) {
    if (tabs.value.includes(tabName)) {
        changeTab(tabName);
    } else {
        tabs.value.push(tabName);
    }
}


defineExpose({ changeTab, addTab, deleteTab, currentTab })
</script>