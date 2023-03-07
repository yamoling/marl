<template>
    <a href="#" @click="fileClicked" @dblclick="() => fileSelected(path)">
        <font-awesome-icon v-if="isDirectory && !expanded" icon="fa-solid fa-folder" />
        <font-awesome-icon v-else-if="isDirectory && expanded" icon="fa-regular fa-folder-open" />
        <font-awesome-icon v-else icon="fa-regular fa-file" />
        {{ fileName }}
    </a>
    <ul v-show="expanded">
        <li v-for="child in childFiles.slice(0, 100)">
            <FileExplorer :path="child.path" :is-directory="child.isDirectory"
                @file-selected="(path) => fileSelected(path)">
            </FileExplorer>
        </li>
        <li v-if="childFiles.length > 100"> {{ childFiles.length - 100 }} file not shown </li>
    </ul>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { HTTP_URL } from '../constants';

interface File {
    path: string
    isDirectory: boolean
}

const expanded = ref(false);
const childFiles = ref([] as File[]);
const props = defineProps<File>();
const emits = defineEmits(["fileSelected"]);
const fileName = computed(() => {
    const splits = props.path.split('/');
    return splits.at(-1);
});

function fileClicked() {
    if (props.isDirectory) {
        if (!expanded.value) {
            loadChildFiles();
        }
        expanded.value = !expanded.value;
    } else {
        fileSelected(props.path);
    }
}

function fileSelected(path: string) {
    emits('fileSelected', path);
}

function loadChildFiles() {
    fetch(`${HTTP_URL}/ls/${props.path}`)
        .then(resp => resp.json())
        .then(dirContent => childFiles.value = dirContent);
}

</script>
