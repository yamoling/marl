<template>
    <div ref="hover" class="hover">
        <ul>
            <li v-for="run in runs"> {{ run.rundir }}: {{ run.progress * 100 }}% </li>
        </ul>

    </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import {Run} from "../../models/Run";


const runs = ref([] as Run[]);
const hover = ref({} as HTMLDivElement)


function show(newRuns: Run[], x: number, y: number) {
    runs.value = newRuns;
    hover.value.style.left = `${x}px`;
    hover.value.style.top = `${y}px`;
    hover.value.style.display = 'block';
}

document.addEventListener('click', () => {
    hover.value.style.display = 'none';
});

defineExpose({ show });

</script>

<style scoped>

.hover {
    width: fit-content;
    position: fixed;
    display: none;
    background-color: #fff;
    border: 1px solid #ccc;
    padding: 5px;
    z-index: 1000;
}

.hover ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.hover ul li {
    padding: 5px 10px;
    cursor: pointer;
}

.hover ul li:hover {
    background-color: #f0f0f0;
}
</style>
