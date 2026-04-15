import { type ReplayTrack } from './replayTimeline';

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