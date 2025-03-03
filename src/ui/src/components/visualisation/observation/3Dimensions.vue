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
        <!-- <table v-if="settingsStore.getExtraViewMode() == 'table'" class="table table-responsive">
            <tbody>
                <tr>
                    <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
                </tr>
                <tr>
                    <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
                </tr>
            </tbody>
        </table> -->
        <!-- <template v-else-if="settingsStore.getExtraViewMode() == 'colour'"> -->
        Min value: {{ Math.min(...extras).toFixed(3) }} <br>
        Max value: {{ Math.max(...extras).toFixed(3) }}
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
import { useSettingsStore } from '../../../stores/SettingsStore';
import { ref } from 'vue';


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

const settingsStore = useSettingsStore();
const rainbow = new Rainbow()
rainbow.setNumberRange(Math.min(...props.extras), Math.max(...props.extras));
rainbow.setSpectrum("blue", "white", "red");
const rows = ref([
    new RowDef("Extras", 0, 5),
    new RowDef("Extras", 5, 4, false),
    new RowDef("Extras", 9, 4, false),
    new RowDef("Extras", 13, 4, false),
    new RowDef("Extras", 17, 4, false),
]);

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