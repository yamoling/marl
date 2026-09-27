/**
 * Settings: replay rules (port of the old v2 logic), timeline track kinds, default statistic and
 * default x axis. Persisted in `localStorage["marl-studio.settings"]`.
 */
import { defineStore } from "pinia";
import { ref, watch } from "vue";
import {
  defaultSettings,
  parseSettings,
  resolveReplay,
  SETTINGS_KEY,
  trainerName,
  type ReplayResolution,
  type Settings,
} from "../domain/settings";
import { parseJSON, readStorage, writeStorage } from "./storage";

export const useSettingsStore = defineStore("settings", () => {
  const settings = ref<Settings>(parseSettings(parseJSON(readStorage(SETTINGS_KEY))));
  watch(settings, (s) => writeStorage(SETTINGS_KEY, JSON.stringify(s)), { deep: true });

  /** Replay rule applying to an experiment (from its raw `experiment.json`). */
  function replayFor(raw: Record<string, unknown> | null | undefined): ReplayResolution {
    return resolveReplay(settings.value, trainerName(raw));
  }

  function setTrainerRule(key: string, onlySavedActions: boolean): void {
    const k = key.trim();
    if (k) settings.value.replay.trainerRules = { ...settings.value.replay.trainerRules, [k]: onlySavedActions };
  }

  function removeTrainerRule(key: string): void {
    const { [key]: _removed, ...rest } = settings.value.replay.trainerRules;
    settings.value.replay.trainerRules = rest;
  }

  function reset(): void {
    settings.value = defaultSettings();
  }

  return { settings, replayFor, setTrainerRule, removeTrainerRule, reset };
});
