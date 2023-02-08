<template>
    <div class="row">
        <div class="col-2 scrollable-list" ref="testList">
            <h4> Testing </h4>
            <font-awesome-icon v-if="replayStore.loadingTests" icon="fa-solid fa-spinner" />
            <div class="accordion">
                <div v-for="(test, testNum) in replayStore.testingList" class="accordion-item">
                    <h4 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                            :data-bs-target="'#' + test.filename" aria-expanded="true" :aria-controls="test.filename"
                            @click="() => test.episodes.forEach((_, e) => replayStore.loadTestEpisodeMetrics(testNum, e))">
                            {{ test.filename }}
                        </button>
                    </h4>
                    <div :id="test.filename" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th> File </th>
                                    <th> Score </th>
                                    <th> Length </th>
                                </tr>
                                <tr v-for="(e, episodeNum) in test.episodes">
                                    <td>
                                        <a href="#" @click="() => onTestEpisodeClicked(testNum, episodeNum)">
                                            {{ e }}
                                        </a>
                                    </td>
                                    <td>
                                        {{ replayStore.testEpisodeMetrics?.[testNum]?.[episodeNum]?.score }}
                                    </td>
                                    <td>
                                        {{ replayStore.testEpisodeMetrics?.[testNum]?.[episodeNum]?.episode_length }}
                                    </td>
                                </tr>
                            </table>
                            </div>
                            </div>
                            </div>
                            </div>
                            </div>
        <div class="col-9">
            <font-awesome-icon v-if="replayStore.loadingMetrics" icon="fa-solid fa-spinner" />
            <Score @episode-selected="onPlotClicked"></Score>
        </div>
    </div>
</template>

<script setup lang="ts">
import Score from '../charts/Score.vue';
import { useEpisodeStore } from '../../stores/EpisodeStore';
import { ref } from 'vue';
import { Collapse } from "bootstrap/dist/js/bootstrap.bundle.min.js";


interface Collapsable {
    collapse: () => void,
    toggle: () => void,
    show: () => void
}



const testList = ref(null as null | HTMLElement);
const replayStore = useEpisodeStore();

function onPlotClicked(kind: string, index: number) {
    console.log(kind, index);
    const id = `div:nth-child(${index + 1}) > div.collapse`;
    const elem = testList.value?.querySelector(id) as HTMLElement | null;
    // emits('episodeSelected', kind, index);
    const selectedAccordeon = new Collapse(elem) as Collapsable;
    console.log(selectedAccordeon);
    selectedAccordeon?.show();
    elem?.parentElement?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
    replayStore.testingList[index].episodes.forEach((_, episodeNum) => {
        replayStore.loadTestEpisodeMetrics(index, episodeNum);
    });
}


function onTestEpisodeClicked(testNum: number, episodeNum: number) {
    console.log(testNum, episodeNum)
    emits('episodeSelected', testNum, episodeNum);
}

const emits = defineEmits(["episodeSelected"])

</script>

<style>
.scrollable-list {
    max-height: 80vh;
    overflow-y: scroll;
}
</style>