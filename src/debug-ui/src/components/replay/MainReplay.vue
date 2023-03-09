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

async function showModal(episode: Promise<ReplayEpisode>) {
    viewingEpisode.value = null;
    modal.show();
    try {
        viewingEpisode.value = await episode;
    } catch (e) {
        alert("Failed to load episode");
    }
}

onMounted(() => {
    modal = new Modal("#episodeViewer");
})



</script>
