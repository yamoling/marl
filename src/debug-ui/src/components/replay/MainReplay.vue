<template>
    <div>
        <Overview @request-view-episode="showModal" />
        <EpisodeViewer :episode="viewingEpisode" :frames="[]" id="episodeViewer" />
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Overview from './Overview.vue';
import { Modal } from "bootstrap";
import EpisodeViewer from './EpisodeViewer.vue';
import { ReplayEpisode } from '../../models/Episode';



const viewingEpisode = ref(null as ReplayEpisode | null);
let modal = {} as Modal;

function showModal(episode: ReplayEpisode) {
    viewingEpisode.value = episode;
    modal.show();
}

onMounted(() => {
    modal = new Modal("#episodeViewer");
})



</script>
