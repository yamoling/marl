<template>
    <!-- <div class="row">
        <div class="col-auto text-center" ref="trainList">
            <h4> Training </h4>
            <table class="table table-sm table-striped table-hover text-center">
                <thead style="position: sticky; top: 0; z-index: 1;">
                    <th class="px-1"> # Episode </th>
                    <th class="px-1"> Length</th>
                    <th class="px-1"> Score </th>
                    <th class="px-1"> Gems </th>
                    <th class="px-1"> Elevator</th>
                </thead>
                <tbody style="overflow-y: auto; height: 100px;">
                    <tr v-for="train in globalState.experiment?.train" @click="() => onTrainEpisodeClicked(train)">
                        <td> {{ train.name }} </td>
                        <td> {{ train.metrics.episode_length }}</td>
                        <td> {{ train.metrics.score }} </td>
                        <td> {{ train.metrics.gems_collected }}</td>
                        <td> {{ train.metrics.in_elevator }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
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
    </div> -->
    <div>
        <Overview @request-view-episode="showModal" />
        <EpisodeViewer :episode="globalState.viewingEpisode" :frames="[]" id="episodeViewer" />
    </div>
</template>

<script setup lang="ts">
import { useReplayStore } from '../../stores/ReplayStore';
import { ref } from 'vue';
import { Collapse } from "bootstrap/dist/js/bootstrap.bundle.min.js";
import { useGlobalState } from '../../stores/GlobalState';
import Overview from '../Overview.vue';
import { Modal } from "bootstrap";
import EpisodeViewer from './EpisodeViewer.vue';


interface Collapsable {
    collapse: () => void,
    toggle: () => void,
    show: () => void
}



const testList = ref(null as null | HTMLElement);
const replayStore = useReplayStore();
const globalState = useGlobalState();


function showModal() {
    new Modal("#episodeViewer").show();
}

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


</script>
