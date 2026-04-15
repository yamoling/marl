import { defineStore } from 'pinia';
import { ref } from 'vue';
import { parseTrackId, ReplayTrackOption, ReplayTrackSelection } from '../components/visualisation/replayTimeline';
import { type TimelineTrackKind } from '../models/Timeline';

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

    function updateTrackKind(logdir: string, trackId: string, kind: TimelineTrackKind, availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
        const parsed = parseTrackId(trackId);
        if (parsed == null) {
            return resolveSelectedTracks(logdir, availableOptions);
        }

        const currentSelections = resolveSelectedTracks(logdir, availableOptions);
        const selectionIndex = currentSelections.findIndex((selection) => selection.key === parsed.key);
        if (selectionIndex < 0) {
            return currentSelections;
        }

        const selection = currentSelections[selectionIndex];
        const componentIndex = selection.componentPaths.findIndex((path) => arePathsEqual(path, parsed.path));
        if (componentIndex < 0) {
            return currentSelections;
        }

        const nextSelections = currentSelections.map((entry, index) => {
            if (index !== selectionIndex) return entry;

            const componentKinds = entry.componentKinds.slice();
            componentKinds[componentIndex] = kind;
            return {
                ...entry,
                componentKinds,
            };
        });

        return setSelectedTracks(logdir, nextSelections, availableOptions);
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
        updateTrackKind,
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
        componentKinds: option.components.map(() => 'numeric' as const),
    }));
}

function sanitizeSelectedTracks(selections: ReplayTrackSelection[], availableOptions: ReplayTrackOption[]): ReplayTrackSelection[] {
    const optionByKey = new Map(availableOptions.map((option) => [option.key, option] as const));
    const nextSelections: ReplayTrackSelection[] = [];

    for (const selection of selections) {
        const option = optionByKey.get(selection.key);
        if (option == null) continue;

        const validEntries = selection.componentPaths
            .map((path, index) => ({ path, kind: normalizeTrackKind(selection.componentKinds[index]) }))
            .filter((entry, index, entries) => entries.findIndex((candidate) => arePathsEqual(candidate.path, entry.path)) === index)
            .filter((entry) => option.components.some((component) => arePathsEqual(component.path, entry.path)));

        if (validEntries.length === 0) continue;

        nextSelections.push({
            key: selection.key,
            alias: selection.alias?.trim() ? selection.alias.trim() : null,
            componentPaths: validEntries.map((entry) => entry.path),
            componentKinds: validEntries.map((entry) => entry.kind),
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

        const rawComponentKinds = (candidate as { componentKinds?: unknown }).componentKinds;
        const componentKinds = Array.isArray(rawComponentKinds)
            ? rawComponentKinds.filter((value): value is TimelineTrackKind => value === 'numeric' || value === 'categorical')
            : [];

        const componentPaths = candidate.componentPaths.filter((path): path is number[] => Array.isArray(path) && path.every((value) => typeof value === 'number' && Number.isInteger(value)));
        return {
            key: candidate.key,
            alias: typeof candidate.alias === 'string' && candidate.alias.trim().length > 0 ? candidate.alias.trim() : null,
            componentPaths,
            componentKinds,
        };
    }).filter((entry): entry is ReplayTrackSelection => entry != null);
}

function arePathsEqual(left: number[], right: number[]): boolean {
    if (left.length !== right.length) return false;

    return left.every((value, index) => value === right[index]);
}

function normalizeTrackKind(kind: TimelineTrackKind | undefined): TimelineTrackKind {
    return kind === 'categorical' ? 'categorical' : 'numeric';
}