<template>
    <Tabs ref="tabs" :tabs="tabNames" />
    <TrainingConfig v-show="tabs.currentTab == 'Config'" @start="onConfigDone" />
    <Trainer v-show="tabs.currentTab == 'Training'" :algorithm="config.algo" :wrappers="config.wrappers"
    :level="config.level" />
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Trainer from './Trainer.vue';
import Tabs from '../Tabs.vue';
import type { ITabs } from "../Tabs.vue";
import TrainingConfig from './TrainingConfig.vue';


const tabNames = ["Config", "Training"] as const;
const tabs = ref({} as ITabs<typeof tabNames>);
const config = ref({ algo: "", level: "", wrappers: [] as string[] });



function onConfigDone(algo: string, level: string, wrappers: string[]) {
    config.value.algo = algo;
    config.value.level = level;
    config.value.wrappers = wrappers;
    tabs.value.changeTab("Training");
}

</script>