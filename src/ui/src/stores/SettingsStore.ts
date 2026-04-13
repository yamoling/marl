import { defineStore } from "pinia";
import { ref } from "vue";
import { Settings } from "../models/Settings";
import { MetricSelection } from "../models/Settings";

export const useSettingsStore = defineStore("SettingsStore", () => {

    const settings = ref(initSettingsFromLocalStorage());

    function initSettingsFromLocalStorage(): Settings {
        const saved = localStorage.getItem("settings");
        if (saved != null) {
            return JSON.parse(saved) as Settings;
        }
        return {
            selectedMetrics: [],
            extrasViewMode: "colour"
        };
    }

    function saveSettingsToLocalStorage() {
        localStorage.setItem("settings", JSON.stringify(settings.value));
    }

    function getSelectedMetrics() {
        return settings.value.selectedMetrics;
    }

    function clearSelectedMetrics() {
        settings.value.selectedMetrics = [];
        saveSettingsToLocalStorage();
    }

    function addSelectedMetric(label: string, category: string) {
        const selection: MetricSelection = { label, category };
        if (!settings.value.selectedMetrics.some(m => m.label === label && m.category === category)) {
            settings.value.selectedMetrics.push(selection);
            saveSettingsToLocalStorage();
        }
    }

    function removeSelectedMetric(label: string, category: string) {
        settings.value.selectedMetrics = settings.value.selectedMetrics.filter((m) => m.label !== label || m.category !== category);
        saveSettingsToLocalStorage();
    }

    function setExtrasViewMode(mode: "table" | "colour") {
        settings.value.extrasViewMode = mode;
        saveSettingsToLocalStorage();
    }

    function getExtraViewMode() {
        return settings.value.extrasViewMode;
    }

    return {
        clearSelectedMetrics,
        getSelectedMetrics,
        addSelectedMetric,
        removeSelectedMetric,
        setExtrasViewMode,
        getExtraViewMode
    };
});
