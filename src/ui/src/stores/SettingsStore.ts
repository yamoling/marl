import { defineStore } from "pinia";
import { ref } from "vue";
import { MetricSelection } from "../models/Settings";

export const useSettingsStore = defineStore("SettingsStore", () => {

    const metrics = ref(initSettingsFromLocalStorage());

    function initSettingsFromLocalStorage(): MetricSelection[] {
        const saved = localStorage.getItem("settings");
        if (saved != null) {
            return JSON.parse(saved);
        }
        return [];
    }

    function saveSettingsToLocalStorage() {
        localStorage.setItem("settings", JSON.stringify(metrics.value));
    }

    function getSelectedMetrics(): MetricSelection[] {
        return metrics.value;
    }

    function clearSelectedMetrics() {
        metrics.value = [];
        saveSettingsToLocalStorage();
    }

    function addSelectedMetric(label: string, category: string) {
        const selection: MetricSelection = { label, category };
        if (!metrics.value.some(m => m.label === label && m.category === category)) {
            metrics.value.push(selection);
            saveSettingsToLocalStorage();
        }
    }

    function removeSelectedMetric(label: string, category: string) {
        metrics.value = metrics.value.filter((m) => m.label !== label || m.category !== category);
        saveSettingsToLocalStorage();
    }

    return {
        clearSelectedMetrics,
        getSelectedMetrics,
        addSelectedMetric,
        removeSelectedMetric,
    };
});
