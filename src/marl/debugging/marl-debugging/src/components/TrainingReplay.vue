<template>
    <h3>Training replay</h3>
    <Tabs @tab-change="changeTab"></Tabs>
    <div id="tab-content">
        <Overview v-show="currentTab == 'Overview'" @episode-selected="selectEpisode"></Overview>
        <EpisodeViewer v-show="currentTab == 'Inspect'" :episode-num="selectedEpisodeNum" :kind="selectedKind">
        </EpisodeViewer>
    </div>

</template>


<script setup lang="ts">
import { ref } from 'vue';
import Tabs from './Tabs.vue';
import EpisodeViewer from './tabs/EpisodeViewer.vue';
import Overview from './tabs/Overview.vue';

const currentTab = ref("Overview" as "Overview" | "Inspect");
const selectedEpisodeNum = ref(0);
const selectedKind = ref("test" as "test" | "train");

function changeTab(newTab: "Overview" | "Inspect") {
    currentTab.value = newTab;
}

function selectEpisode(kind: "test" | "train", episodeNum: number) {
    currentTab.value = "Inspect";
    selectedEpisodeNum.value = episodeNum;
    selectedKind.value = kind;
    console.log(kind, episodeNum);
}

</script>

<style>

</style>