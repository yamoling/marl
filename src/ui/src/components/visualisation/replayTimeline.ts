import { ActionDetails, ReplayEpisode } from '../../models/Episode';
import { TimelineTrack, type TimelineTrackKind } from '../../models/Timeline';
import { formatNumber } from './numberFormat';
import {
    arePathsEqual,
    defaultTrackSelections,
    normalizeTrackKind,
    ReplayTrackComponentOption,
    ReplayTrackOption,
    ReplayTrackSelection,
    sanitizeTrackSelections,
} from './replayTimelineSelection';

export type ReplayTrack = TimelineTrack;

type DetailLeafKind = 'numeric' | 'categorical';
type DetailShapeKind = 'scalar' | 'vector' | 'matrix';

type DetailShape = {
    kind: DetailShapeKind;
    leafKind: DetailLeafKind;
    dimensions: number[];
};

type DetailSeries = {
    path: number[];
    label: string;
};

export function buildReplayTracks(
    replayEpisode: ReplayEpisode | null,
    rewardValues: number[],
    nAgents: number,
    selectedTracks: ReplayTrackSelection[] | null = null,
): ReplayTrack[] {
    const tracks = [] as ReplayTrack[];

    tracks.push(new TimelineTrack('reward', 'Reward', 'numeric', rewardValues));

    if (replayEpisode == null || nAgents <= 0) {
        return tracks;
    }

    const availableOptions = discoverReplayTrackOptions(replayEpisode, nAgents);
    const nextSelections = selectedTracks == null
        ? defaultTrackSelections(availableOptions)
        : sanitizeTrackSelections(selectedTracks, availableOptions);

    for (const selection of nextSelections) {
        const option = availableOptions.find((entry) => entry.key === selection.key);
        if (option == null) continue;

        const selectedComponents = selection.componentPaths
            .map((path, index) => {
                const component = option.components.find((entry) => arePathsEqual(entry.path, path));
                if (component == null) return null;

                return {
                    component,
                    kind: normalizeTrackKind(selection.componentKinds[index]),
                };
            })
            .filter((component): component is { component: ReplayTrackComponentOption; kind: TimelineTrackKind } => component != null);

        if (selectedComponents.length === 0) continue;

        tracks.push(...buildTracksForDetailSelection(replayEpisode.action_details, option, selection, selectedComponents));
    }

    return tracks;
}

export function discoverReplayTrackOptions(replayEpisode: ReplayEpisode | null, nAgents: number): ReplayTrackOption[] {
    if (replayEpisode == null) return [];

    return collectDetailKeys(replayEpisode.action_details).map((key) => {
        const shape = inferDetailShape(replayEpisode.action_details, key);
        const components = shape == null ? [] : buildSeriesDescriptors(key, shape, nAgents);
        return {
            key,
            label: formatDetailKeyLabel(key),
            trackCount: components.length,
            trackLabels: components.map((entry) => entry.label),
            components,
        } satisfies ReplayTrackOption;
    }).filter((option) => option.trackCount > 0);
}

export function currentTrackValueLabelAtStep(track: ReplayTrack, currentStep: number): string {
    if (currentStep <= 0) return '-';

    const index = currentStep - 1;
    if (index < 0 || index >= track.values.length) return '-';

    const value = track.values[index];
    if (track.kind === 'numeric') {
        const numericValue = typeof value === 'number'
            ? value
            : (typeof value === 'string' ? Number.parseFloat(value) : 0);
        return formatNumber(numericValue);
    }

    return value == null ? 'none' : String(value);
}

function buildTracksForDetailSelection(
    details: ActionDetails[],
    option: ReplayTrackOption,
    selection: ReplayTrackSelection,
    selectedComponents: Array<{ component: ReplayTrackComponentOption; kind: TimelineTrackKind }>,
): ReplayTrack[] {
    const baseLabel = selection.alias?.trim().length ? selection.alias.trim() : option.label;

    return selectedComponents.map(({ component, kind }) => {
        const values = details.map((detail) => normalizeSeriesValue(extractPathValue(detail[option.key], component.path), kind));
        if (!values.some((value) => value != null)) return null;

        const label = component.label === baseLabel
            ? baseLabel
            : `${baseLabel} - ${component.label}`;

        return new TimelineTrack(
            buildTrackId(option.key, component.path),
            label,
            kind,
            values,
        );
    }).filter((track): track is ReplayTrack => track != null);
}

function collectDetailKeys(details: ActionDetails[]): string[] {
    const orderedKeys: string[] = [];
    const seenKeys = new Set<string>();

    for (const detail of details) {
        for (const key of Object.keys(detail)) {
            if (seenKeys.has(key)) continue;
            seenKeys.add(key);
            orderedKeys.push(key);
        }
    }

    return orderedKeys;
}

function buildSeriesDescriptors(key: string, shape: DetailShape, nAgents: number): DetailSeries[] {
    const baseLabel = formatDetailKeyLabel(key);

    if (shape.kind === 'scalar') {
        return [{ path: [], label: baseLabel }];
    }

    if (shape.kind === 'vector') {
        return Array.from({ length: shape.dimensions[0] }, (_, index) => ({
            path: [index],
            label: `${baseLabel} ${formatSeriesIndexLabel(index, shape.dimensions[0], nAgents)}`,
        }));
    }

    return Array.from({ length: shape.dimensions[0] }, (_, rowIndex) => Array.from({ length: shape.dimensions[1] }, (_, columnIndex) => ({
        path: [rowIndex, columnIndex],
        label: `${baseLabel} ${formatSeriesIndexLabel(rowIndex, shape.dimensions[0], nAgents)} / ${columnIndex + 1}`,
    }))).flat();
}

function inferDetailShape(details: ActionDetails[], key: string): DetailShape | null {
    let shape: DetailShape | null = null;

    for (const detail of details) {
        const candidate = inspectDetailValue(detail[key]);
        if (candidate == null) continue;

        if (shape == null) {
            shape = candidate;
            continue;
        }

        if (candidate.kind === 'matrix' || (candidate.kind === 'vector' && shape.kind === 'scalar')) {
            shape = candidate;
            continue;
        }

        if (shape.kind === candidate.kind) {
            shape = {
                kind: shape.kind,
                leafKind: shape.leafKind === 'categorical' || candidate.leafKind === 'categorical' ? 'categorical' : 'numeric',
                dimensions: shape.dimensions.map((dimension, index) => Math.max(dimension, candidate.dimensions[index] ?? 0)),
            };
            continue;
        }

        if (shape.kind === 'vector' && candidate.kind === 'scalar') {
            shape = {
                kind: shape.kind,
                leafKind: shape.leafKind === 'categorical' || candidate.leafKind === 'categorical' ? 'categorical' : 'numeric',
                dimensions: shape.dimensions,
            };
        }
    }

    return shape;
}

function inspectDetailValue(value: unknown): DetailShape | null {
    if (value == null) return null;

    if (!Array.isArray(value)) {
        return {
            kind: 'scalar',
            leafKind: isNumericLike(value) ? 'numeric' : 'categorical',
            dimensions: [],
        };
    }

    if (value.length === 0) {
        return {
            kind: 'vector',
            leafKind: 'categorical',
            dimensions: [0],
        };
    }

    if (value.some((item) => Array.isArray(item))) {
        const rows = value.filter((item): item is unknown[] => Array.isArray(item));
        const maxColumns = rows.reduce((max, row) => Math.max(max, row.length), 0);
        return {
            kind: 'matrix',
            leafKind: inferLeafKind(rows.flat()),
            dimensions: [rows.length, maxColumns],
        };
    }

    return {
        kind: 'vector',
        leafKind: inferLeafKind(value),
        dimensions: [value.length],
    };
}

function inferLeafKind(values: unknown[]): DetailLeafKind {
    return values.every((value) => isNumericLike(value)) ? 'numeric' : 'categorical';
}

function isNumericLike(value: unknown): boolean {
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value === 'boolean') return true;
    if (typeof value === 'string') {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed);
    }
    return false;
}

function normalizeSeriesValue(value: unknown, leafKind: DetailLeafKind): number | string | null {
    if (leafKind === 'numeric') {
        return normalizeNumeric(value);
    }

    return normalizeCategorical(value);
}

function normalizeCategorical(value: unknown): string | null {
    if (value == null) return null;
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return String(value);
    }
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

function normalizeNumeric(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'boolean') return value ? 1 : 0;
    if (typeof value === 'string') {
        const parsed = Number.parseFloat(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return null;
}

function extractPathValue(value: unknown, path: number[]): unknown {
    let current: unknown = value;

    for (const index of path) {
        if (!Array.isArray(current) || index < 0 || index >= current.length) {
            return null;
        }

        current = current[index];
    }

    return current;
}

function buildTrackId(key: string, path: number[]): string {
    return `detail:${key}:${path.length === 0 ? 'scalar' : path.join('.')}`;
}

function formatDetailKeyLabel(key: string): string {
    switch (key) {
        case 'q_values':
            return 'Q-values';
        case 'action_probabilities':
            return 'Action probabilities';
        case 'options':
            return 'Options';
        case 'options_termination_probs':
            return 'Option termination probabilities';
        default:
            return key
                .replaceAll('_', ' ')
                .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
                .split(' ')
                .filter((part) => part.length > 0)
                .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
                .join(' ');
    }
}

function formatSeriesIndexLabel(index: number, total: number, nAgents: number): string {
    if (total === nAgents && nAgents > 0) {
        return `Agent ${index + 1}`;
    }

    return `${index + 1}`;
}
