<template>
    <div class="row">
        <h3> Layers </h3>
        <table v-for="layer in obs" :style="{ width: `${layer.length * 10}px` }" class="m-1">
            <tr v-for="row in layer">
                <td v-for="item in row" class="grid-item" :style="{ backgroundColor: getColour(item) }">
                </td>
            </tr>
        </table>

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


function getColour(value: number) {
    if (value == 1) return "red";
    if (value == -1) return "blue";
    if (value == 0) return "white";
    return "black";
}

defineProps<{
    obs: number[][][],
    extras: number[]
}>();

</script>
<style>
.grid-item {
    width: 10px;
    height: 10px;
    border: solid 1px;
}
</style>