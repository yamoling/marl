<template>
    <div>
        <TrainingConfig v-if="!globalState.logdir" />
        <Trainer ref="trainer" v-show="globalState.logdir != null" />
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useGlobalState } from '../../stores/GlobalState';
import Trainer from './Trainer.vue';
import TrainingConfig from './TrainingConfig.vue';

const globalState = useGlobalState();
const trainer = ref({} as typeof Trainer);

</script>