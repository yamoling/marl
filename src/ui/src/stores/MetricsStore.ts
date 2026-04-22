import { defineStore } from "pinia";
import { ref } from "vue";
import { MetricSelection, MetricSelectionSchema } from "../models/Metrics";
import { useErrorStore } from "./ErrorStore";

const LOCAL_STORAGE_KEY = "metrics";

export const useMetricsStore = defineStore("MetricsStore", () => {
  const errorStore = useErrorStore();
  const metrics = ref(load());

  function load(): MetricSelection[] {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (raw == null) {
      return [];
    }
    try {
      const result = MetricSelectionSchema.array().safeParse(JSON.parse(raw));
      if (result.success) {
        return result.data;
      }
    } catch {
      // JSON.parse failed or unexpected error
    }
    errorStore.push(
      "Metrics selection reset",
      "Your saved metrics selection could not be read and has been reset to defaults.",
    );
    return [];
  }

  function saveSettingsToLocalStorage() {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(metrics.value));
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
    if (
      !metrics.value.some((m) => m.label === label && m.category === category)
    ) {
      metrics.value.push(selection);
      saveSettingsToLocalStorage();
    }
  }

  function removeSelectedMetric(label: string, category: string) {
    metrics.value = metrics.value.filter(
      (m) => m.label !== label || m.category !== category,
    );
    saveSettingsToLocalStorage();
  }

  return {
    clearSelectedMetrics,
    getSelectedMetrics,
    addSelectedMetric,
    removeSelectedMetric,
  };
});
