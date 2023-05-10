<template>
    <div class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Experiment details {{ experiment.logdir }} </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body row">
                    <pre id="someId" ref="someRef" class="col"></pre>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { ExperimentInfo } from '../../models/Infos';


const someRef = ref({} as HTMLElement);
const props = defineProps<{
    experiment: ExperimentInfo
}>();


onMounted(() => {
    someRef.value.innerHTML = syntaxHighlight(props.experiment);
});

function syntaxHighlight(json: object | string) {
    if (typeof json != 'string') {
        json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replaceAll("\\n", "<br>");
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}


</script>
<style>
pre {
    outline: 1px solid #ccc;
    padding: 5px;
    margin: 5px;
}

.string {
    color: green;
}

.number {
    color: darkorange;
}

.boolean {
    color: blue;
}

.null {
    color: magenta;
}

.key {
    color: red;
}
</style>