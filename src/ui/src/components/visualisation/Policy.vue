<template>
    <table class="table table-sm table-hover table-striped">
        <thead>
            <tr>
                <th colspan="5">Action probabilities</th>
            </tr>
            <tr>
                <td v-for="action in ACTION_MEANINGS"> {{ action }}</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td v-for="(_, i) in ACTION_MEANINGS"> {{ policy[i] }} </td>
            </tr>
        </tbody>
    </table>
</template>

<script setup lang="ts">
import { computed } from 'vue';


const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"] as const;
const POLICIES = ["Softmax", "EpsilonGreedy"] as const;

const props = defineProps<{
    qvalues: number[],
    policy: typeof POLICIES[number],
}>();

const policy = computed(() => {
    if (props.policy == "Softmax") {
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