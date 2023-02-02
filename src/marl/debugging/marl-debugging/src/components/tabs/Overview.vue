<template>
    <div class="row">
        <div class="col-3 row">
            <div class="col-6 scrollable-list">
                Training
                <ul>
                    <li v-for="training in replayStore.trainingList"> <a href="#"> {{ training }} </a></li>
                </ul>
            </div>

            <div class="col-6 scrollable-list">
                Testing
                <ul>
                    <li v-for="test in replayStore.testingList"> <a href="#"> {{ test }} </a></li>
                </ul>
            </div>
        </div>
        <div class="col-9">
            <Score @episode-selected="(k, e) => emits('episodeSelected', k, e)"></Score>
        </div>
    </div>
</template>

<script setup lang="ts">
import Score from '../charts/Score.vue';
import { useEpisodeStore } from '../../stores/EpisodeStore';


const replayStore = useEpisodeStore();
const emits = defineEmits(["episodeSelected"])
</script>

<style>
.scrollable-list {
    max-height: 80vh;
    overflow-y: scroll;
}
</style>