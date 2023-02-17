import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";

export const useAlgorithmStore = defineStore("AlgorithmStore", () => {

    const algorithms = ref([] as string[]);
    const envWrappers = ref([] as string[]);
    const maps = ref([] as string[]);

    function refresh() {
        fetch(`${HTTP_URL}/algo/list`)
            .then(resp => resp.json())
            .then(algoList => algorithms.value = algoList);

        fetch(`${HTTP_URL}/env/wrapper/list`)
            .then(resp => resp.json())
            .then(envWrapperList => envWrappers.value = envWrapperList);
        fetch(`${HTTP_URL}/env/maps/list`)
            .then(resp => resp.json())
            .then(levelList => maps.value = levelList);
    }

    refresh();


    return { algorithms, envWrappers, maps };
});
