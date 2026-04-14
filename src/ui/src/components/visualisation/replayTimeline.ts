import { ReplayEpisode } from '../../models/Episode';
import { TimelineTrack } from '../../models/Timeline';
import { formatNumber } from './numberFormat';

export type ReplayTrack = TimelineTrack;

export function buildReplayTracks(
    replayEpisode: ReplayEpisode | null,
    rewardValues: number[],
    nAgents: number,
): ReplayTrack[] {
    const tracks = [] as ReplayTrack[];

    tracks.push(new TimelineTrack('reward', 'Reward', 'numeric', rewardValues));

    if (replayEpisode == null || nAgents <= 0) {
        return tracks;
    }

    for (let agentNum = 0; agentNum < nAgents; agentNum++) {
        const optionValues = rewardValues.map((_, stepIndex) => {
            const option = replayEpisode.action_details[stepIndex]?.options?.[agentNum];
            return normalizeOption(option);
        });
        if (optionValues.some((option) => option != null)) {
            tracks.push(new TimelineTrack(
                `options-agent-${agentNum + 1}`,
                `Option A${agentNum + 1}`,
                'categorical',
                optionValues,
            ));
        }

        const terminationProbabilities = rewardValues.map((_, stepIndex) => {
            const value = replayEpisode.action_details[stepIndex]?.options_termination_probs?.[agentNum];
            return normalizeContinuous(value);
        });
        if (terminationProbabilities.some((value) => value != null)) {
            tracks.push(new TimelineTrack(
                `termination-prob-agent-${agentNum + 1}`,
                `Option term. A${agentNum + 1}`,
                'numeric',
                terminationProbabilities.map((value) => value ?? 0),
            ));
        }
    }

    return tracks;
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

export function loadTimelineOrder(storageKey: string): string[] | null {
    if (typeof window === 'undefined') return null;

    const raw = window.localStorage.getItem(storageKey);
    if (raw == null) return null;

    try {
        const parsed = JSON.parse(raw) as Partial<{ order: unknown }>;
        if (!Array.isArray(parsed.order)) return null;
        return parsed.order.filter((entry): entry is string => typeof entry === 'string');
    } catch {
        return null;
    }
}

export function persistTimelineOrder(storageKey: string, order: string[]): void {
    if (typeof window === 'undefined') return;

    window.localStorage.setItem(storageKey, JSON.stringify({ order }));
}

export function syncTimelineOrder(currentOrder: string[], tracks: ReplayTrack[], storedOrder: string[] | null): string[] {
    const trackIds = tracks.map((track) => track.id);
    const nextOrderSource = storedOrder ?? currentOrder;
    const nextOrder = nextOrderSource.filter((trackId) => trackIds.includes(trackId));

    for (const trackId of trackIds) {
        if (!nextOrder.includes(trackId)) nextOrder.push(trackId);
    }

    return nextOrder;
}

function normalizeOption(value: unknown): string | null {
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

function normalizeContinuous(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string') {
        const parsed = Number.parseFloat(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return null;
}