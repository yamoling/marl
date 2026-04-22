import { defineStore } from "pinia";
import { ref } from "vue";
import type { TrackConfig } from "../models/Timeline";
import type { Experiment } from "../models/Experiment";
import {
    createDefaultSettings,
    normalizeSettings,
    resolveReplaySettings,
    resolveTrackRule,
    type TrackRule,
    type ReplaySettings,
    type UserSettings,
    type VisualizationSettings,
} from "../models/Settings";

const STORAGE_KEY = "ui_settings";

function loadSettings(): UserSettings {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw == null) {
        return createDefaultSettings();
    }

    try {
        return normalizeSettings(JSON.parse(raw));
    } catch {
        return createDefaultSettings();
    }
}

function persistSettings(settings: UserSettings) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

function normalizeKey(key: string) {
    return key.trim();
}

function updateRuleMap(current: Record<string, boolean>, key: string, value: boolean) {
    const normalizedKey = normalizeKey(key);
    if (normalizedKey.length === 0) {
        return current;
    }

    return {
        ...current,
        [normalizedKey]: value,
    };
}

export const useSettingsStore = defineStore("SettingsStore", () => {
    const settings = ref(loadSettings());

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

    function removeColour(logdir: string) {
        const nextColours = { ...settings.value.visualization.colours };
        delete nextColours[logdir];
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                colours: nextColours,
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

    function setSelectedTracks(logdir: string, tracks: TrackConfig[]) {
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                selectedTracksByLogdir: {
                    ...settings.value.visualization.selectedTracksByLogdir,
                    [logdir]: tracks.map((track) => ({ label: track.label, kind: track.kind })),
                },
            },
        };
        save();
    }

    function removeSelectedTracks(logdir: string) {
        const nextSelected = { ...settings.value.visualization.selectedTracksByLogdir };
        delete nextSelected[logdir];
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                selectedTracksByLogdir: nextSelected,
            },
        };
        save();
    }

    function setDefaultTrackKind(trackLabel: string, kind: "numeric" | "categorical") {
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
        const nextDefaultKinds = { ...settings.value.visualization.tracks.defaultKinds };
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

    function setGlobalTrackRule(trackLabel: string, rule: TrackRule) {
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    globalRules: {
                        ...settings.value.visualization.tracks.globalRules,
                        [trackLabel]: rule,
                    },
                },
            },
        };
        save();
    }

    function removeGlobalTrackRule(trackLabel: string) {
        const nextGlobalRules = { ...settings.value.visualization.tracks.globalRules };
        delete nextGlobalRules[trackLabel];
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    globalRules: nextGlobalRules,
                },
            },
        };
        save();
    }

    function setTrainerTrackRule(trainerName: string, trackLabel: string, rule: TrackRule) {
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    trainerRules: {
                        ...settings.value.visualization.tracks.trainerRules,
                        [trainerName]: {
                            ...(settings.value.visualization.tracks.trainerRules[trainerName] ?? {}),
                            [trackLabel]: rule,
                        },
                    },
                },
            },
        };
        save();
    }

    function removeTrainerTrackRule(trainerName: string, trackLabel: string) {
        const trainerRules = { ...(settings.value.visualization.tracks.trainerRules[trainerName] ?? {}) };
        delete trainerRules[trackLabel];
        const nextTrainerRules = { ...settings.value.visualization.tracks.trainerRules };
        if (Object.keys(trainerRules).length === 0) {
            delete nextTrainerRules[trainerName];
        } else {
            nextTrainerRules[trainerName] = trainerRules;
        }
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    trainerRules: nextTrainerRules,
                },
            },
        };
        save();
    }

    function setExperimentTrackRule(logdir: string, trackLabel: string, rule: TrackRule) {
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    experimentRules: {
                        ...settings.value.visualization.tracks.experimentRules,
                        [logdir]: {
                            ...(settings.value.visualization.tracks.experimentRules[logdir] ?? {}),
                            [trackLabel]: rule,
                        },
                    },
                },
            },
        };
        save();
    }

    function removeExperimentTrackRule(logdir: string, trackLabel: string) {
        const experimentRules = { ...(settings.value.visualization.tracks.experimentRules[logdir] ?? {}) };
        delete experimentRules[trackLabel];
        const nextExperimentRules = { ...settings.value.visualization.tracks.experimentRules };
        if (Object.keys(experimentRules).length === 0) {
            delete nextExperimentRules[logdir];
        } else {
            nextExperimentRules[logdir] = experimentRules;
        }
        settings.value = {
            ...settings.value,
            visualization: {
                ...settings.value.visualization,
                tracks: {
                    ...settings.value.visualization.tracks,
                    experimentRules: nextExperimentRules,
                },
            },
        };
        save();
    }

    function setGlobalOnlySavedActions(onlySavedActions: boolean) {
        settings.value = {
            ...settings.value,
            replay: {
                ...settings.value.replay,
                globalOnlySavedActions: onlySavedActions,
            },
        };
        save();
    }

    function setTrainerReplayRule(trainerName: string, onlySavedActions: boolean) {
        settings.value = {
            ...settings.value,
            replay: {
                ...settings.value.replay,
                trainerRules: updateRuleMap(settings.value.replay.trainerRules, trainerName, onlySavedActions),
            },
        };
        save();
    }

    function removeTrainerReplayRule(trainerName: string) {
        const normalizedKey = normalizeKey(trainerName);
        if (normalizedKey.length === 0) {
            return;
        }

        const nextRules = { ...settings.value.replay.trainerRules };
        delete nextRules[normalizedKey];
        settings.value = {
            ...settings.value,
            replay: {
                ...settings.value.replay,
                trainerRules: nextRules,
            },
        };
        save();
    }

    function setExperimentReplayRule(logdir: string, onlySavedActions: boolean) {
        settings.value = {
            ...settings.value,
            replay: {
                ...settings.value.replay,
                experimentRules: updateRuleMap(
                    settings.value.replay.experimentRules,
                    logdir,
                    onlySavedActions,
                ),
            },
        };
        save();
    }

    function removeExperimentReplayRule(logdir: string) {
        const normalizedKey = normalizeKey(logdir);
        if (normalizedKey.length === 0) {
            return;
        }

        const nextRules = { ...settings.value.replay.experimentRules };
        delete nextRules[normalizedKey];
        settings.value = {
            ...settings.value,
            replay: {
                ...settings.value.replay,
                experimentRules: nextRules,
            },
        };
        save();
    }

    function resetReplaySettings() {
        replace(createDefaultSettings());
    }

    function resetSettings() {
        replace(createDefaultSettings());
    }

    function resolveTrackPreference(
        experiment: Pick<Experiment, "logdir" | "trainer"> | null,
        trackLabel: string,
        fallbackKind: "numeric" | "categorical" = "numeric",
    ) {
        return resolveTrackRule(settings.value, experiment, trackLabel, fallbackKind);
    }

    function getSelectedTracks(logdir: string) {
        return settings.value.visualization.selectedTracksByLogdir[logdir] ?? [];
    }

    function resolveReplay(experiment: Pick<Experiment, "logdir" | "trainer"> | null) {
        return resolveReplaySettings(settings.value, experiment);
    }

    return {
        settings,
        setReplaySettings,
        setVisualizationSettings,
        setColour,
        removeColour,
        clearColours,
        setSelectedTracks,
        removeSelectedTracks,
        setDefaultTrackKind,
        removeDefaultTrackKind,
        setGlobalTrackRule,
        removeGlobalTrackRule,
        setTrainerTrackRule,
        removeTrainerTrackRule,
        setExperimentTrackRule,
        removeExperimentTrackRule,
        setGlobalOnlySavedActions,
        setTrainerReplayRule,
        removeTrainerReplayRule,
        setExperimentReplayRule,
        removeExperimentReplayRule,
        resetReplaySettings,
        resetSettings,
        resolveTrackPreference,
        getSelectedTracks,
        resolveReplay,
    };
});
