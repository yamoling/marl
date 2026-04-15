import { defineStore } from 'pinia';
import { ref } from 'vue';
import { ReplayTrackOption, ReplayTrackSelection } from '../components/visualisation/replayTimeline';

type ReplayTimelineSelectionMap = Record<string, unknown>;

const STORAGE_KEY = 'replayTimelineSelections';

export const useReplayTimelineStore = defineStore('ReplayTimelineStore', () => {
    const selections = ref(loadSelections());

    function resolveSelectedTracks(logdir: string, availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
        const storedSelection = selections.value[logdir];
        if (storedSelection == null) {
            return defaultSelections(availableOptions);
        }

        if (Array.isArray(storedSelection) && storedSelection.every((entry) => typeof entry === 'string')) {
            const legacyKeys = storedSelection.filter((entry): entry is string => typeof entry === 'string');
            return defaultSelections(availableOptions).filter((selection) => legacyKeys.includes(selection.key));
        }

        const parsed = parseSelections(storedSelection);
        return sanitizeSelectedTracks(parsed, availableOptions);
    }

    function setSelectedTracks(logdir: string, nextSelections: ReplayTrackSelection[], availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
        const sanitizedSelections = sanitizeSelectedTracks(nextSelections, availableOptions);
        selections.value = {
            ...selections.value,
            [logdir]: sanitizedSelections,
        };
        saveSelections(selections.value);
        return sanitizedSelections;
    }

    function clearSelectedTracks(logdir: string) {
        if (!(logdir in selections.value)) return;

        const nextSelections = { ...selections.value };
        delete nextSelections[logdir];
        selections.value = nextSelections;
        saveSelections(selections.value);
    }

    return {
        clearSelectedTracks,
        resolveSelectedTracks,
        setSelectedTracks,
    };
});

function loadSelections(): ReplayTimelineSelectionMap {
    if (typeof window === 'undefined') return {};

    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (raw == null) return {};

        const parsed = JSON.parse(raw) as Record<string, unknown>;
        const nextSelections: ReplayTimelineSelectionMap = {};

        for (const [logdir, value] of Object.entries(parsed)) {
            nextSelections[logdir] = value;
        }

        return nextSelections;
    } catch {
        return {};
    }
}

function saveSelections(selections: ReplayTimelineSelectionMap): void {
    if (typeof window === 'undefined') return;

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(selections));
}

function defaultSelections(availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
    return availableOptions.map((option) => ({
        key: option.key,
        alias: null,
        componentPaths: option.components.map((component) => component.path),
    }));
}

function sanitizeSelectedTracks(selections: ReplayTrackSelection[], availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
    const optionByKey = new Map(availableOptions.map((option) => [option.key, option] as const));
    const nextSelections: ReplayTrackSelection[] = [];

    for (const selection of selections) {
        const option = optionByKey.get(selection.key);
        if (option == null) continue;

        const uniquePaths = selection.componentPaths.filter((path, index, paths) => paths.findIndex((candidate) => arePathsEqual(candidate, path)) === index);
        const validPaths = uniquePaths.filter((path) => option.components.some((component) => arePathsEqual(component.path, path)));
        if (validPaths.length === 0) continue;

        nextSelections.push({
            key: selection.key,
            alias: selection.alias?.trim() ? selection.alias.trim() : null,
            componentPaths: validPaths,
        });
    }

    return nextSelections;
}

function parseSelections(value: unknown): ReplayTrackSelection[] {
    if (!Array.isArray(value)) return [];

    return value.map((entry): ReplayTrackSelection | null => {
        if (entry == null || typeof entry !== 'object') return null;

        const candidate = entry as Partial<ReplayTrackSelection> & { componentPaths?: unknown };
        if (typeof candidate.key !== 'string') return null;
        if (!Array.isArray(candidate.componentPaths)) return null;

        const componentPaths = candidate.componentPaths.filter((path): path is number[] => Array.isArray(path) && path.every((value) => typeof value === 'number' && Number.isInteger(value)));
        return {
            key: candidate.key,
            alias: typeof candidate.alias === 'string' && candidate.alias.trim().length > 0 ? candidate.alias.trim() : null,
            componentPaths,
        };
    }).filter((entry): entry is ReplayTrackSelection => entry != null);
}

function arePathsEqual(left: number[], right: number[]): boolean {
    if (left.length !== right.length) return false;

    return left.every((value, index) => value === right[index]);
}