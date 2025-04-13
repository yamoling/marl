import { defineStore } from "pinia";
import { ref } from "vue";
import { Qvalues } from "../models/Qvalues";

export const useQvaluesStore = defineStore("QvaluesStore", () => {

    const qvalues_settings = ref(initQvaluesFromLocalStorage());

    function initQvaluesFromLocalStorage(): Qvalues {
        const saved = localStorage.getItem("qvalues_settings");
        if (saved != null) {
            return JSON.parse(saved) as Qvalues;
        }
        return {
            selectedQvalues: [],
            extrasViewMode: "colour"
        };
    }

    function saveQvaluesToLocalStorage() {
        localStorage.setItem("qvalues_settings", JSON.stringify(qvalues_settings.value));
    }

    function getSelectedQvalues() {
        return qvalues_settings.value.selectedQvalues;
    }

    function clearSelectedQvalues() {
        qvalues_settings.value.selectedQvalues = [];
        saveQvaluesToLocalStorage();
    }

    function addSelectedQvalue(qvalue: string) {
        qvalues_settings.value.selectedQvalues.push(qvalue);
        saveQvaluesToLocalStorage();
    }

    function removeSelectedQvalue(qvalue: string) {
        qvalues_settings.value.selectedQvalues = qvalues_settings.value.selectedQvalues.filter((m) => m !== qvalue);
        saveQvaluesToLocalStorage();
    }

    function setExtrasViewMode(mode: "table" | "colour") {
        qvalues_settings.value.extrasViewMode = mode;
        saveQvaluesToLocalStorage();
    }

    function getExtraViewMode() {
        console.log(qvalues_settings.value.extrasViewMode);
        return qvalues_settings.value.extrasViewMode;
    }

    return {
        clearSelectedQvalues,
        getSelectedQvalues,
        addSelectedQvalue,
        removeSelectedQvalue,
        setExtrasViewMode,
        getExtraViewMode
    };
});
