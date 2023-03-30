<template>
    <tr>
        <th> Probas </th>
        <td v-for="(_, i) in ACTION_MEANINGS"> {{ probs[i].toFixed(3) }} </td>
    </tr>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { ACTION_MEANINGS, POLICIES } from "../../constants";



const props = defineProps<{
    qvalues: number[],
    policy: typeof POLICIES[number],
}>();

const probs = computed(() => {
    if (props.policy == "SoftmaxPolicy") {
        return softmax(props.qvalues);
    } else if (props.policy == "EpsilonGreedy") {
        return epsilonGreedy(props.qvalues);
    } else {
        return [];
    }
});

function softmax(qvalues: number[], tau: number = 1.0): number[] {
    // Exponentiate the qvalues
    const expQ = qvalues.map(q => Math.exp(q / tau));
    // Sum the exponentiated qvalues
    const sumExpQ = expQ.reduce((a, b) => a + b, 0);
    // Divide each row by the sum of the row
    return expQ.map(q => q / sumExpQ);
}

function epsilonGreedy(qvalues: number[], epsilon: number = 0.1): number[] {
    const maxIndex = qvalues.indexOf(Math.max(...qvalues));
    const epsilonProb = epsilon / (qvalues.length - 1);
    return qvalues.map((_, i) => {
        if (i == maxIndex) {
            return 1 - epsilon;
        }
        return epsilonProb;
    });
}

</script>