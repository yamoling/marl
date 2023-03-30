<template>
    <div class="row">
        <h3> Layers </h3>
        <div class="col-auto m-2" v-for="layer in obs">
            <div class="row" v-for="row in layer">
                <div v-for="col in row" style="border: solid 1px; width: 20px;"> {{ col }}</div>
            </div>
            <br>
        </div>

        <h3> Extras </h3>
        <table class="table table-responsive">
            <tr>
                <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
            </tr>
            <tr>
                <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
            </tr>
        </table>
    </div>
</template>

<script setup lang="ts">

const layerNames = computed(() => {
    const names = [];
    for (let i = 0; i < props.envInfo.n_agents; i++) {
        names.push(`Agent ${i}`);
    }
    names.push("Walls");
    for (let i = 0; i < props.envInfo.n_agents; i++) {
        names.push(`Laser ${i}`);
    }
    names.push("Gems");
    names.push("Elevator");
    return names;
});

defineProps<{
    obs: number[][][],
    extras: number[]
}>();

</script>