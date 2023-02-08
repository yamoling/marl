<template>
    <h3>Training replay</h3>
    <Tabs ref="tabs" @tab-change="onTabChanged"></Tabs>
    <div id="tab-content">
        <Overview v-show="currentTab == 'Overview'" @episode-selected="selectEpisode"></Overview>
        <EpisodeViewer v-show="currentTab == 'Inspect'" ref="inspect">
        </EpisodeViewer>
    </div>

</template>


<script setup lang="ts">
import { ref } from 'vue';
import Tabs from './Tabs.vue';
import EpisodeViewer from './tabs/EpisodeViewer.vue';
import Overview from './tabs/Overview.vue';

interface Tabs {
    changeTab: (newTab: "Overview" | "Inspect") => void
}

const tabs = ref(null as Tabs | null);

interface EpisodeViewerInterface {
    setEpisode: (step: number, episodeNum: number) => void
}

const currentTab = ref("Overview" as "Overview" | "Inspect");
const inspect = ref(null as EpisodeViewerInterface | null);

function onTabChanged(newTab: "Overview" | "Inspect") {
    currentTab.value = newTab;
}

function selectEpisode(stepNum: number, episodeNum: number) {
    inspect.value?.setEpisode(stepNum, episodeNum);
    tabs.value?.changeTab('Inspect');
}

</script>

<style>

</style>