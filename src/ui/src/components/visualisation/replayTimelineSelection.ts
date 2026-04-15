import { type TimelineTrackKind } from '../../models/Timeline';

export interface ReplayTrackComponentOption {
    path: number[];
    label: string;
}

export interface ReplayTrackOption {
    key: string;
    label: string;
    trackCount: number;
    trackLabels: string[];
    components: ReplayTrackComponentOption[];
}

export interface ReplayTrackSelection {
    key: string;
    alias: string | null;
    componentPaths: number[][];
    componentKinds: TimelineTrackKind[];
}

export function defaultTrackSelections(options: ReplayTrackOption[]): ReplayTrackSelection[] {
    return options.map((option) => ({
        key: option.key,
        alias: null,
        componentPaths: option.components.map((component) => component.path),
        componentKinds: option.components.map(() => 'numeric' as const),
    }));
}

export function sanitizeTrackSelections(selections: ReplayTrackSelection[], options: ReplayTrackOption[]): ReplayTrackSelection[] {
    const optionByKey = new Map(options.map((option) => [option.key, option] as const));
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

export function parseTrackId(trackId: string): { key: string; path: number[] } | null {
    if (!trackId.startsWith('detail:')) return null;

    const encoded = trackId.slice('detail:'.length);
    const separatorIndex = encoded.lastIndexOf(':');
    if (separatorIndex < 0) return null;

    const key = encoded.slice(0, separatorIndex);
    const suffix = encoded.slice(separatorIndex + 1);
    if (key.length === 0) return null;

    if (suffix === 'scalar') {
        return { key, path: [] };
    }

    const path = suffix.split('.').map((part) => Number.parseInt(part, 10));
    if (path.some((part) => Number.isNaN(part) || part < 0)) return null;

    return { key, path };
}

export function arePathsEqual(left: number[], right: number[]): boolean {
    if (left.length !== right.length) return false;

    return left.every((value, index) => value === right[index]);
}

export function normalizeTrackKind(kind: string | TimelineTrackKind | undefined): TimelineTrackKind {
    return kind === 'categorical' ? 'categorical' : 'numeric';
}