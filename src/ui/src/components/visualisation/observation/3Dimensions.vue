<template>
    <div class="row">
        <h3> Layers </h3>
        <table v-for="layer in obs" :style="{ width: `${layer.length * 10}px` }" class="m-1">
            <tbody>
                <tr v-for="row in layer">
                    <td v-for="item in row" class="grid-item" :style="{ backgroundColor: getColour(item) }">
                    </td>
                </tr>
            </tbody>
        </table>

        <h3> Extras </h3>
        <table class="m-1">
            <tbody>
                <template v-for="row in rows">
                    <tr v-if="row.showMeanings">
                        <td v-for="meaning in row.meanings()" class="grid-item">
                            {{ meaning }}
                        </td>
                    </tr>
                    <tr>
                        <td v-for="extra in row.values()" class="grid-item"
                            :style="{ backgroundColor: `#${rainbow.colourAt(extra)}` }">
                            {{ extra.toFixed(3) }}
                        </td>
                    </tr>
                </template>
            </tbody>
        </table>
        <!-- </template> -->
    </div>
</template>

<script setup lang="ts">
import Rainbow from 'rainbowvis.js';
import { computed } from 'vue';


class RowDef {
    public title: string
    public startIndex: number
    public nItems: number
    public showMeanings: boolean

    constructor(title: string, startIndex: number, nItems: number, showMeanings = true) {
        this.title = title;
        this.startIndex = startIndex;
        this.nItems = nItems;
        this.showMeanings = showMeanings;
    }

    public indices() {
        return Array(this.nItems).keys().map(i => i + this.startIndex);
    }

    public values() {
        return this.indices().map(i => props.extras[i]);
    }

    public meanings() {
        return this.indices().map(i => props.extrasMeanings[i]);
    }

    public items(): [number, string][] {
        const indices = this.indices();
        return Array.from(indices.map(i => [props.extras[i], props.extrasMeanings[i]]))
    }
}

const props = defineProps<{
    obs: number[][][],
    extras: number[],
    extrasMeanings: string[]
}>();

const rainbow = new Rainbow()
rainbow.setNumberRange(Math.min(...props.extras), Math.max(...props.extras));
rainbow.setSpectrum("blue", "white", "red");
const rows = computed(() => {
    let i = 0;
    const res = []
    while (i < props.extras.length) {
        const rowDef = new RowDef("Extras", i, Math.min(props.extras.length - i, 4), true);
        i += rowDef.nItems;
        res.push(rowDef);
    }
    return res;
});


function getColour(value: number) {
    if (value == 1) return "red";
    if (value == -1) return "blue";
    if (value == 0) return "white";
    return "black";
}



</script>
<style>
.grid-item {
    width: 10px;
    height: 10px;
    border: solid 1px;
}
</style>