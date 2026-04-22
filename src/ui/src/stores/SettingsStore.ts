import { defineStore } from "pinia";
import { ref } from "vue";
import type { Experiment } from "../models/Experiment";
import {
  createDefaultSettings,
  parseSettings,
  resolveReplaySettings,
  resolveTrackRule,
  type ReplaySettings,
  type UserSettings,
  type VisualizationSettings,
} from "../models/Settings";
import { useErrorStore } from "./ErrorStore";

const STORAGE_KEY = "ui_settings";

function persistSettings(settings: UserSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export const useSettingsStore = defineStore("SettingsStore", () => {
  const errorStore = useErrorStore();
  const settings = ref(loadSettings());

  function loadSettings(): UserSettings {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw == null) {
      return createDefaultSettings();
    }

    try {
      const parsed = parseSettings(JSON.parse(raw));
      if (parsed != null) {
        return parsed;
      }
    } catch {
      // JSON.parse failed — fall through to reset
    }

    errorStore.push(
      "Settings reset",
      "Your saved settings could not be read and have been reset to defaults.",
    );
    return createDefaultSettings();
  }

  function save() {
    persistSettings(settings.value);
  }

  function replace(next: UserSettings) {
    settings.value = next;
    save();
  }

  function setReplaySettings(replay: ReplaySettings) {
    replace({
      ...settings.value,
      replay,
    });
  }

  function setVisualizationSettings(visualization: VisualizationSettings) {
    replace({
      ...settings.value,
      visualization,
    });
  }

  function setColour(logdir: string, colour: string) {
    settings.value = {
      ...settings.value,
      visualization: {
        ...settings.value.visualization,
        colours: {
          ...settings.value.visualization.colours,
          [logdir]: colour,
        },
      },
    };
    save();
  }

  function clearColours() {
    settings.value = {
      ...settings.value,
      visualization: {
        ...settings.value.visualization,
        colours: {},
      },
    };
    save();
  }

  function setDefaultTrackKind(
    trackLabel: string,
    kind: "numeric" | "categorical",
  ) {
    settings.value = {
      ...settings.value,
      visualization: {
        ...settings.value.visualization,
        tracks: {
          ...settings.value.visualization.tracks,
          defaultKinds: {
            ...settings.value.visualization.tracks.defaultKinds,
            [trackLabel]: kind,
          },
        },
      },
    };
    save();
  }

  function removeDefaultTrackKind(trackLabel: string) {
    const nextDefaultKinds = {
      ...settings.value.visualization.tracks.defaultKinds,
    };
    delete nextDefaultKinds[trackLabel];
    settings.value = {
      ...settings.value,
      visualization: {
        ...settings.value.visualization,
        tracks: {
          ...settings.value.visualization.tracks,
          defaultKinds: nextDefaultKinds,
        },
      },
    };
    save();
  }

  function resetSettings() {
    replace(createDefaultSettings());
  }

  function resolveTrackPreference(
    trackLabel: string,
    fallbackKind: "numeric" | "categorical" = "numeric",
  ) {
    return resolveTrackRule(settings.value, trackLabel, fallbackKind);
  }

  function resolveReplay(
    experiment: Pick<Experiment, "logdir" | "trainer"> | null,
  ) {
    return resolveReplaySettings(settings.value, experiment);
  }

  return {
    settings,
    setReplaySettings,
    setVisualizationSettings,
    setColour,
    clearColours,
    setDefaultTrackKind,
    removeDefaultTrackKind,
    resetSettings,
    resolveTrackPreference,
    resolveReplay,
  };
});
