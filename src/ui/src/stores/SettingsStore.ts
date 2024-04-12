import { defineStore } from "pinia";
import { ref } from "vue";
import { Settings } from "../models/Settings";

export const useSettingsStore = defineStore("SettingsStore", () => {

    const settings = ref(initSettingsFromLocalStorage());

    function initSettingsFromLocalStorage(): Settings {
        const saved = localStorage.getItem("settings");
        if (saved != null) {
            return JSON.parse(saved) as Settings;
        }
        return {
            selectedMetrics: []
        };
    }

    function saveSettingsToLocalStorage() {
        localStorage.setItem("settings", JSON.stringify(settings.value));
    }

    function getSelectedMetrics() {
        return settings.value.selectedMetrics;
    }

    function addSelectedMetric(metric: string) {
        settings.value.selectedMetrics.push(metric);
        saveSettingsToLocalStorage();
    }

    function removeSelectedMetric(metric: string) {
        settings.value.selectedMetrics = settings.value.selectedMetrics.filter((m) => m !== metric);
        saveSettingsToLocalStorage();
    }


    return {
        getSelectedMetrics,
        addSelectedMetric,
        removeSelectedMetric
    };
});
