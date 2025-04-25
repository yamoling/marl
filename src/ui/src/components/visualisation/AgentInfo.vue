<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        Obs shape = {{ obsShape }}
        <OneDimension v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras" :env-info="experiment.env" />
        <ThreeDimension v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras"
            :extras-meanings="extrasMeanings" />
        <p v-else> No preview available for {{ obsDimensions }} dimensions </p>
        <h4> Actions & Qvalues </h4>
        <table class="table table-responsive">
            <thead>
                <tr>
                    <th scope="row"> Actions <br> available </th>
                    <th scope="col" :style="{ opacity: (availableActions[action] == 1) ? 1 : 0.5 }"
                        v-for="(meaning, action) in experiment.env.action_space.action_names">
                        {{ meaning }}
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr v-if="qvalues" v-for="(objective, objectiveNum) in experiment.env.reward_space.labels">
                    <th scope="row" class="text-capitalize"> {{ objective }} </th>
                    <td v-for="action in experiment.env.n_actions"
                        :style='{ "background-color": "#" + backgroundColours[action - 1][objectiveNum] }'>
                        <template v-if="isMultiObjective">
                            {{ multiObjectiveQvalues[action - 1][objectiveNum].toFixed(4) }}
                        </template>
                        <template v-else>
                            {{ singleObjectiveQvalues[action - 1].toFixed(4) }}
                        </template>
                    </td>
                </tr>
                {{ isMultiObjective }}
                <template v-if="logits">
                    <tr v-if="isMultiObjective" v-for="(objective, objectiveNum) in experiment.env.reward_space.labels">
                        TODO
                    </tr>
                    <tr v-else>
                        <th> Logits </th>
                        <td> {{ logits }}</td>
                        <!-- <td v-for="logit in logits" :style='{ "background-color": "#" + rainbow.colourAt(logit) }'>
                            {{ logit.toFixed(4) }}
                        </td> -->
                    </tr>

                </template>
                <!-- <tr v-if="episode?.probs && episode.probs.length > currentStep">
                    <th> <b>Probs</b></th>
                    <td v-for="prob in episode.probs[currentStep][agentNum]">
                        {{ prob[0].toFixed(4) }}
                    </td>
                </tr> -->
                <!-- <tr v-if="episode?.messages && episode.messages.length > currentStep"
                    v-for="(messages, index) in episode.messages[currentStep][0][agentNum]">
                    <th>
                        <b>Message to {{ index }}</b>
                    </th>
                    <td v-for="message in messages">
                        {{ message.toFixed(4) }}
                    </td>
                </tr> -->
                <!-- <tr v-if="episode?.received_messages && episode.received_messages.length > currentStep">
                    <th> <b>Received Messages</b></th>
                    <td v-for="message in episode.received_messages[currentStep][agentNum]">
                        {{ message.toFixed(4) }}
                    </td>
                </tr> -->
                <!-- <tr v-if="episode?.init_qvalues && episode.init_qvalues.length > currentStep">
                    <th> <b>Init Qvalues</b></th>
                    <td v-for="qvalue in episode.init_qvalues[currentStep][agentNum]">
                        {{ qvalue.toFixed(4) }}
                    </td>
                </tr> -->
                <!-- <tr v-if="episode?.messages && episode.messages.length > 0">
                    <th> <b>Messages</b></th>
                    <template v-for="messages in episode.messages[0][currentStep][agentNum]">
                        <td>
                            {{ messages }}
                        </td>
                    </template>
                </tr> -->
            </tbody>
            <tfoot v-if="isMultiObjective && qvalues != null">
                <tr>
                    <td> <b>Q-Total</b></td>
                    <td v-for="action in qvalues.length"
                        :style='{ "background-color": "#" + totalQValuesColours[action - 1] }'>
                        {{ totalQValues[action - 1].toFixed(4) }}
                    </td>
                </tr>
            </tfoot>
        </table>
    </div>
</template>

<script setup lang="ts">

import { computed } from "vue";
import type Rainbow from "rainbowvis.js";
import { ReplayEpisode } from "../../models/Episode";
import ThreeDimension from "./observation/3Dimensions.vue";
import OneDimension from "./observation/1Dimension.vue";
import { Experiment } from "../../models/Experiment";
import { computeShape } from "../../utils";



const props = defineProps<{
    episode: ReplayEpisode | null
    agentNum: number
    currentStep: number
    rainbow: Rainbow
    experiment: Experiment
}>();

const obsShape = computed(() => {
    if (props.episode?.episode == null) return [];
    return computeShape(props.episode.episode.all_observations[0][0])
});
const obsDimensions = computed(() => obsShape.value.length);
const episodeLength = computed(() => {
    if (props.episode == null) return 0;
    return props.episode.frames.length - 1;
});
const isMultiObjective = computed(() => props.experiment.env.reward_space.size > 1);

const obs = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.all_observations[props.currentStep][props.agentNum];
});

const extras = computed(() => {
    if (props.episode == null) return []
    return props.episode.episode.all_extras[props.currentStep][props.agentNum];
});

const extrasMeanings = computed(() => props.experiment.env.extras_meanings)

const availableActions = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.all_available_actions[props.currentStep][props.agentNum];
});

const multiObjectiveQvalues = computed(() => qvalues.value as number[][])

const singleObjectiveQvalues = computed(() => qvalues.value as number[])

const qvalues = computed(() => {
    if (props.episode == null) return null;
    if (props.episode.qvalues == null || props.episode.qvalues.length == 0) return null;
    if (props.currentStep >= episodeLength.value) return null;
    return props.episode.qvalues[props.currentStep][props.agentNum];
});

const totalQValues = computed(() => {
    const res = [] as number[];
    for (let i = 0; i < multiObjectiveQvalues.value.length; i++) {
        let sum = 0;
        for (let j = 0; j < multiObjectiveQvalues.value[i].length; j++) {
            sum += multiObjectiveQvalues.value[i][j];
        }
        res.push(sum);
    }
    return res;
});

const backgroundColours = computed(() => {
    if (qvalues.value == null) return [];
    let actionQvalues = qvalues.value
    if (!isMultiObjective.value) {
        actionQvalues = actionQvalues.map(q => [q]) as number[][];
    }
    const colours = (actionQvalues as number[][]).map(qs => qs.map(q => props.rainbow.colourAt(q)));
    return colours;
});



const totalQValuesColours = computed(() => {
    const colours = totalQValues.value.map(q => props.rainbow.colourAt(q));
    return colours;
});
const multiObjectiveLogits = computed(() => {
    if (props.episode == null) return [];
    if (props.episode.logits == null || props.episode.logits.length == 0) return [];
    if (props.currentStep >= episodeLength.value) return [];
    return props.episode.logits[props.currentStep][props.agentNum] as number[][];
});
const logits = computed(() => {
    if (props.episode == null) return null;
    if (props.episode.logits == null || props.episode.logits.length == 0) return null;
    if (props.currentStep >= episodeLength.value) return null;
    return props.episode.logits[props.currentStep][props.agentNum] as number[];
})

const obsFlattened = computed(() => obs.value as number[]);
const obsLayered = computed(() => obs.value as number[][][]);

</script>


<style>
.agent-info {
    border-radius: 1px;
    border-style: solid;
    border-color: gainsboro;
    border-radius: 2%;
}


tfoot {
    border-top: 2px solid black;
}
</style>