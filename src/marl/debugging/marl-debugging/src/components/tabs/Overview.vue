<template>
    <div class="row">
        <div class="col-3 row">
            <div class="col-6 scrollable-list">
                <h4> Training </h4>
                <ul>
                    <li v-for="training in replayStore.trainingList"> <a href="#"> {{ training }} </a></li>
                </ul>
            </div>

            <div class="col-6 scrollable-list">
                <h4> Testing </h4>
                <div class="accordion" id="accordionExample">
                    <div v-for="test in replayStore.testingList" class="accordion-item">
                        <h4 class="accordion-header" id="headingOne">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                :data-bs-target="'#'+test.name" aria-expanded="true" :aria-controls="test.name">
                                {{ test.name }}
                            </button>
                        </h4>
                        <div :id="test.name" class="accordion-collapse collapse" aria-labelledby="headingOne"
                            data-bs-parent="#accordionExample" style="">
                            <div class="accordion-body">
                                <ul>
                                    <li v-for="episode in test.episodes"> {{ episode }}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
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