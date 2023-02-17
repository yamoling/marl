<template>
    <h3>Replay</h3>
    <Tabs ref="tabs" :tabs="tabNames"></Tabs>
    <div id="tab-content">
        <div v-show="tabs.currentTab == 'File selection'">
            <h4> Select a file </h4>
            <FileExplorer path="logs" :is-directory="true" @file-selected="onReplayFileSelected" />
        </div>
        <Overview v-show="tabs.currentTab == 'Overview'" @episode-selected="selectEpisode"></Overview>
        <EpisodeViewer v-show="tabs.currentTab == 'Inspect'" :episode="episode" :frames="frames">
        </EpisodeViewer>
    </div>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import Tabs from '../Tabs.vue';
import { ITabs } from "../Tabs.vue";
import EpisodeViewer from './EpisodeViewer.vue';
import Overview from './Overview.vue';
import { useEpisodeStore } from '../../stores/EpisodeStore';
import { Episode } from '../../models/Episode';
import FileExplorer from '../FileExplorer.vue';
import { HTTP_URL } from '../../constants';

const tabNames = [
    "File selection",
    "Overview",
    "Inspect"
] as const;

const tabs = ref({} as ITabs<typeof tabNames>);
const store = useEpisodeStore();
const episode = ref(null as Episode | null);
const frames = ref([] as string[]);



function selectEpisode(stepNum: number, episodeNum: number) {
    store.getTestEpisode(stepNum, episodeNum)
        .then(e => {
            console.log(e);
            episode.value = e;
        });
    store.getTestFrames(stepNum, episodeNum)
        .then(newFrames => frames.value = newFrames);
    tabs.value.changeTab('Inspect');
}

function onReplayFileSelected(path: string) {
    fetch(`${HTTP_URL}/load/${path}`)
        .then(() => tabs.value.changeTab('Overview'));
}

</script>

<style>

</style>