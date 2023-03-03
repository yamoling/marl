<template>
    <div>
        <h3>Replay</h3>
        <Tabs ref="tabs" :tabs="tabNames"></Tabs>
        <div id="tab-content">
            <div v-show="tabs.currentTab == 'File selection'">
                <h4> Select a file </h4>
                <FileExplorer path="logs" :is-directory="true" @file-selected="onReplayFolderSelected" />
            </div>
            <Overview v-show="tabs.currentTab == 'Overview'" @test-episode-selected="selectTestEpisode"
                @train-episode-selected="selectTrainEpisode" />
            <EpisodeViewer v-show="tabs.currentTab == 'Inspect'" :episode="episode" :frames="frames" />
        </div>
    </div>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import Tabs from '../Tabs.vue';
import { ITabs } from "../Tabs.vue";
import EpisodeViewer from './EpisodeViewer.vue';
import Overview from './MainReplay.vue';
import { useReplayStore } from '../../stores/ReplayStore';
import type { ReplayEpisode } from '../../models/Episode';
import FileExplorer from '../FileExplorer.vue';
import { HTTP_URL } from '../../constants';

const tabNames = [
    "File selection",
    "Overview",
    "Inspect"
] as const;

const tabs = ref({} as ITabs<typeof tabNames>);
const store = useReplayStore();
const episode = ref(null as ReplayEpisode | null);
const frames = ref([] as string[]);



function selectTestEpisode(stepNum: number, episodeNum: number) {
    store.getTestEpisode(stepNum, episodeNum)
        .then(e => {
            episode.value = e;
        });
    store.getTestFrames(stepNum, episodeNum)
        .then(newFrames => frames.value = newFrames);
    tabs.value.changeTab('Inspect');
}

function selectTrainEpisode(episodeNum: number) {
    store.getTrainEpisode(episodeNum)
        .then(e => episode.value = e);
}

function onReplayFolderSelected(path: string) {
    store.refresh();
    fetch(`${HTTP_URL}/load/${path}`)
        .then(() => tabs.value.changeTab('Overview'));
}

</script>
