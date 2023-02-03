<template>
    <div class="row">
        <div class="col-3 row">
            <div class="col-6 scrollable-list">
                <h4> Training </h4>
                <font-awesome-icon v-if="replayStore.loadingTrain" icon="fa-spin fa-solid fa-spinner" />
                <ul>
                    <li v-for="training in replayStore.trainingList.slice(0, 100)"> <a href="#"> {{ training }} </a></li>
                    <li v-if="replayStore.trainingList.length > 100"> {{ replayStore.trainingList.length - 100 }} items not shown</li>
                </ul>
            </div>

            <div class="col-6 scrollable-list">
                <h4> Testing </h4>
                <font-awesome-icon v-if="replayStore.loadingTests" icon="fa-solid fa-spinner" />
                <div class="accordion">
                    <div v-for="test in replayStore.testingList" class="accordion-item">
                        <h4 class="accordion-header" id="headingOne">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                :data-bs-target="'#'+test" aria-expanded="true" :aria-controls="test">
                                {{ test }}
                            </button>
                        </h4>
                        <div :id="test" class="accordion-collapse collapse" aria-labelledby="headingOne"
                            data-bs-parent="#accordionExample" style="">
                            <div class="accordion-body">
                                <ul>
                                    some text
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-9">
            <font-awesome-icon v-if="replayStore.loadingMetrics" icon="fa-solid fa-spinner" />
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