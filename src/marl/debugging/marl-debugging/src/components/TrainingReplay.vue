<template>
    <h3>Training replay</h3>
    <Tabs @tab-change="changeTab"></Tabs>
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

interface EpisodeViewerInterface {
    setEpisode: (step: number, num: number) => void
}

const currentTab = ref("Overview" as "Overview" | "Inspect");
const inspect = ref(null as EpisodeViewerInterface | null);

function changeTab(newTab: "Overview" | "Inspect") {
    currentTab.value = newTab;
}

function selectEpisode(kind: "test" | "train", episodeNum: number) {
    inspect.value?.setEpisode(episodeNum, 0);
    currentTab.value = "Inspect";
}

</script>

<style>

</style>