<template>
    <table class="table table-responsive">
        <tr>
            <th colspan="5" class="obs">Observation</th>
        </tr>
        <template v-for="row in featureRows">
            <tr>
                <th v-for="feature in row" class="obs"> {{ feature.meaning }} </th>
            </tr>
            <tr>
                <td v-for="feature in row"> {{ (feature.meaning) ? feature.value.toFixed(2) : "" }}</td>
            </tr>
        </template>
        <template v-if="extras.length > 0">
            <tr>
                <th :colspan="extras.length" class="extras"> Extras </th>
            </tr>
            <tr>
                <td class="extras" v-for="e in extras"> {{ e.toFixed(3) }}</td>
            </tr>
        </template>
    </table>
</template>

<script setup lang="ts">
import { computed } from '@vue/reactivity';


const props = defineProps<{
    obs: number[],
    extras: number[]
}>();

const meanings = computed(() => {
    const res = [
        "Gem count",
        "Gem i",
        "Gem j",
        "Finish i",
        "Finish j",
        "Agent count",
        "Closest agent i",
        "Closest agent j",
    ]
    for (let i = 0; i < 4; i++) {
        ["NORTH", "EAST", "SOUTH", "WEST"].forEach(direction => {
            res.push(`${direction}-${i}`);
        })
    };
    let i = res.length;
    while (i < props.obs.length) {
        res.push("Agent x gem i");
        res.push("Agent x gem j");
        res.push("Agent x end i");
        res.push("Agent x end j");
        i += 4;
    }
    return res;
})


const featureRows = computed(() => {
    const rows = [] as { meaning: string, value: number }[][];
    // Five items per row with meaning and value attributes
    for (let i = 0; i < props.obs.length; i += 5) {
        const row = [] as { meaning: string, value: number }[];
        for (let j = 0; j < 5; j++) {
            row.push({
                meaning: meanings.value[i + j] || "",
                value: props.obs[i + j] || 0
            })
        }
        rows.push(row);
    }
    return rows;
});


</script>

<style scoped>
.obs {
    background-color: beige;
}

.extras {
    background-color: whitesmoke;
}
</style>