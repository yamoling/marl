import { z } from "zod";
import type { Experiment } from "./Experiment";
import type { TrackConfig, TimelineTrackKind } from "./Timeline";

export const SETTINGS_SCHEMA_VERSION = 2 as const;

export type ReplayRuleSource = "global" | "trainer" | "experiment";
export type TrackRuleSource = "global" | "trainer" | "experiment" | "default";

export interface ReplaySettings {
    globalOnlySavedActions: boolean;
    trainerRules: Record<string, boolean>;
    experimentRules: Record<string, boolean>;
}

export interface TrackRule {
    visible: boolean;
    kind: TimelineTrackKind;
}

export interface TrackSettings {
    defaultKinds: Record<string, TimelineTrackKind>;
    globalRules: Record<string, TrackRule>;
    trainerRules: Record<string, Record<string, TrackRule>>;
    experimentRules: Record<string, Record<string, TrackRule>>;
}

export interface VisualizationSettings {
    colours: Record<string, string>;
    selectedTracksByLogdir: Record<string, TrackConfig[]>;
    tracks: TrackSettings;
}

export interface UserSettings {
    version: typeof SETTINGS_SCHEMA_VERSION;
    replay: ReplaySettings;
    visualization: VisualizationSettings;
}

export interface ReplayRuleResolution {
    onlySavedActions: boolean;
    source: ReplayRuleSource;
    key: string | null;
}

export interface TrackRuleResolution {
    visible: boolean;
    kind: TimelineTrackKind;
    source: TrackRuleSource;
    key: string | null;
}

function escapeRegex(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function globLikePatternToRegex(pattern: string): RegExp {
    let regex = "^";

    for (let index = 0; index < pattern.length; index += 1) {
        const char = pattern[index];
        if (char === "*") {
            regex += ".*";
            continue;
        }
        if (char === "?") {
            regex += ".";
            continue;
        }
        if (char === "{") {
            const endIndex = pattern.indexOf("}", index + 1);
            if (endIndex > index + 1) {
                const content = pattern.slice(index + 1, endIndex);
                const values = content
                    .split(",")
                    .map((entry) => entry.trim())
                    .filter((entry) => entry.length > 0)
                    .map((entry) => escapeRegex(entry));
                if (values.length > 0) {
                    regex += `(?:${values.join("|")})`;
                    index = endIndex;
                    continue;
                }
            }
        }

        regex += escapeRegex(char);
    }

    regex += "$";
    return new RegExp(regex);
}

function buildRuleMatcher(ruleKey: string): RegExp {
    const normalized = ruleKey.trim();
    if (normalized.startsWith("/") && normalized.lastIndexOf("/") > 0) {
        const lastSlashIndex = normalized.lastIndexOf("/");
        const body = normalized.slice(1, lastSlashIndex);
        const rawFlags = normalized.slice(lastSlashIndex + 1);
        const flags = rawFlags.replace(/[^dgimsuy]/g, "");
        try {
            return new RegExp(body, flags);
        } catch {
            return globLikePatternToRegex(normalized);
        }
    }

    return globLikePatternToRegex(normalized);
}

export function matchesTrackRuleKey(ruleKey: string, trackLabel: string): boolean {
    const normalizedKey = ruleKey.trim();
    if (normalizedKey.length === 0) {
        return false;
    }

    if (normalizedKey === trackLabel) {
        return true;
    }

    return buildRuleMatcher(normalizedKey).test(trackLabel);
}

function resolveTrackKindFromDefaultKinds(
    defaultKinds: Record<string, TimelineTrackKind>,
    trackLabel: string,
    fallbackKind: TimelineTrackKind,
): { key: string | null; kind: TimelineTrackKind } {
    const exactKind = defaultKinds[trackLabel];
    if (exactKind != null) {
        return { key: trackLabel, kind: exactKind };
    }

    let matchedKey: string | null = null;
    let matchedKind: TimelineTrackKind | null = null;
    for (const [ruleKey, kind] of Object.entries(defaultKinds)) {
        if (!matchesTrackRuleKey(ruleKey, trackLabel)) {
            continue;
        }
        if (matchedKey == null || ruleKey.length > matchedKey.length) {
            matchedKey = ruleKey;
            matchedKind = kind;
        }
    }

    return {
        key: matchedKey,
        kind: matchedKind ?? fallbackKind,
    };
}

const ReplayRuleMapSchema = z.record(z.string(), z.boolean());
const TrackKindSchema = z.enum(["numeric", "categorical"]);

const TrackRuleSchema = z.object({
    visible: z.boolean(),
    kind: TrackKindSchema,
});

const ReplaySettingsSchema = z.object({
    globalOnlySavedActions: z.boolean(),
    trainerRules: ReplayRuleMapSchema,
    experimentRules: ReplayRuleMapSchema,
});

const TrackSettingsSchema = z.object({
    defaultKinds: z.record(z.string(), TrackKindSchema),
    globalRules: z.record(z.string(), TrackRuleSchema),
    trainerRules: z.record(z.string(), z.record(z.string(), TrackRuleSchema)),
    experimentRules: z.record(z.string(), z.record(z.string(), TrackRuleSchema)),
});

const VisualizationSettingsSchema = z.object({
    colours: z.record(z.string(), z.string()),
    selectedTracksByLogdir: z.record(
        z.string(),
        z.array(
            z.object({
                label: z.string(),
                kind: TrackKindSchema,
            }),
        ),
    ),
    tracks: TrackSettingsSchema,
});

const UserSettingsSchema = z.object({
    version: z.literal(SETTINGS_SCHEMA_VERSION),
    replay: ReplaySettingsSchema,
    visualization: VisualizationSettingsSchema,
});

const LegacyV1SettingsSchema = z.object({
    version: z.literal(1),
    replay: ReplaySettingsSchema,
});

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value != null && !Array.isArray(value);
}

function readRuleMap(raw: unknown): Record<string, boolean> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextRules: Record<string, boolean> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (typeof value === "boolean") {
            nextRules[key] = value;
            continue;
        }

        if (isRecord(value) && typeof value.onlySavedActions === "boolean") {
            nextRules[key] = value.onlySavedActions;
        }
    }

    return nextRules;
}

function readTrackRuleMap(raw: unknown): Record<string, TrackRule> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextRules: Record<string, TrackRule> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (!isRecord(value)) {
            continue;
        }

        const visible = typeof value.visible === "boolean" ? value.visible : null;
        const kind = value.kind === "numeric" || value.kind === "categorical" ? value.kind : null;
        if (visible == null || kind == null) {
            continue;
        }

        nextRules[key] = { visible, kind };
    }

    return nextRules;
}

function readNestedTrackRuleMap(raw: unknown): Record<string, Record<string, TrackRule>> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextRules: Record<string, Record<string, TrackRule>> = {};
    for (const [scopeKey, scopeValue] of Object.entries(raw)) {
        nextRules[scopeKey] = readTrackRuleMap(scopeValue);
    }

    return nextRules;
}

function readTrackKindMap(raw: unknown): Record<string, TimelineTrackKind> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextKinds: Record<string, TimelineTrackKind> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (value === "numeric" || value === "categorical") {
            nextKinds[key] = value;
        }
    }

    return nextKinds;
}

function readColourMap(raw: unknown): Record<string, string> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextColours: Record<string, string> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (typeof value === "string") {
            nextColours[key] = value;
        }
    }

    return nextColours;
}

function readSelectedTracksMap(raw: unknown): Record<string, TrackConfig[]> {
    if (!isRecord(raw)) {
        return {};
    }

    const nextSelected: Record<string, TrackConfig[]> = {};
    for (const [logdir, value] of Object.entries(raw)) {
        if (!Array.isArray(value)) {
            continue;
        }

        const tracks: TrackConfig[] = [];
        for (const entry of value) {
            if (!isRecord(entry)) {
                continue;
            }
            if (typeof entry.label !== "string") {
                continue;
            }
            if (entry.kind !== "numeric" && entry.kind !== "categorical") {
                continue;
            }
            tracks.push({ label: entry.label, kind: entry.kind });
        }
        nextSelected[logdir] = tracks;
    }

    return nextSelected;
}

function createDefaultVisualizationSettings(): VisualizationSettings {
    return {
        colours: {},
        selectedTracksByLogdir: {},
        tracks: {
            defaultKinds: {},
            globalRules: {},
            trainerRules: {},
            experimentRules: {},
        },
    };
}

export function createDefaultSettings(): UserSettings {
    return {
        version: SETTINGS_SCHEMA_VERSION,
        replay: {
            globalOnlySavedActions: false,
            trainerRules: {},
            experimentRules: {},
        },
        visualization: createDefaultVisualizationSettings(),
    };
}

export function normalizeSettings(raw: unknown): UserSettings {
    const parsed = UserSettingsSchema.safeParse(raw);
    if (parsed.success) {
        return parsed.data;
    }

    const legacyParsed = LegacyV1SettingsSchema.safeParse(raw);
    if (legacyParsed.success) {
        return {
            version: SETTINGS_SCHEMA_VERSION,
            replay: legacyParsed.data.replay,
            visualization: createDefaultVisualizationSettings(),
        };
    }

    if (!isRecord(raw)) {
        return createDefaultSettings();
    }

    const replay = isRecord(raw.replay) ? raw.replay : null;
    const visualization = isRecord(raw.visualization) ? raw.visualization : null;
    const visualizationTracks: Record<string, unknown> | null =
        visualization != null && isRecord(visualization.tracks) ? visualization.tracks : null;

    return {
        version: SETTINGS_SCHEMA_VERSION,
        replay: {
            globalOnlySavedActions:
                typeof replay?.globalOnlySavedActions === "boolean"
                    ? replay.globalOnlySavedActions
                    : false,
            trainerRules: readRuleMap(replay?.trainerRules),
            experimentRules: readRuleMap(replay?.experimentRules),
        },
        visualization: {
            colours: readColourMap(visualization?.colours),
            selectedTracksByLogdir: readSelectedTracksMap(visualization?.selectedTracksByLogdir),
            tracks: {
                defaultKinds: readTrackKindMap(visualizationTracks?.defaultKinds),
                globalRules: readTrackRuleMap(visualizationTracks?.globalRules),
                trainerRules: readNestedTrackRuleMap(visualizationTracks?.trainerRules),
                experimentRules: readNestedTrackRuleMap(visualizationTracks?.experimentRules),
            },
        },
    };
}

export function resolveTrackRule(
    settings: UserSettings,
    experiment: Pick<Experiment, "logdir" | "trainer"> | null,
    trackLabel: string,
    fallbackKind: TimelineTrackKind = "numeric",
): TrackRuleResolution {
    const trainerName =
        experiment != null && typeof experiment.trainer?.name === "string"
            ? experiment.trainer.name.trim()
            : "";
    const logdir = experiment?.logdir?.trim() ?? "";

    const experimentRule =
        logdir.length > 0 ? settings.visualization.tracks.experimentRules[logdir]?.[trackLabel] : undefined;
    if (experimentRule != null) {
        return {
            ...experimentRule,
            source: "experiment",
            key: logdir,
        };
    }

    const trainerRule =
        trainerName.length > 0 ? settings.visualization.tracks.trainerRules[trainerName]?.[trackLabel] : undefined;
    if (trainerRule != null) {
        return {
            ...trainerRule,
            source: "trainer",
            key: trainerName,
        };
    }

    const globalRule = settings.visualization.tracks.globalRules[trackLabel];
    if (globalRule != null) {
        return {
            ...globalRule,
            source: "global",
            key: trackLabel,
        };
    }

    const defaultKind = resolveTrackKindFromDefaultKinds(
        settings.visualization.tracks.defaultKinds,
        trackLabel,
        fallbackKind,
    );

    return {
        visible: true,
        kind: defaultKind.kind,
        source: "default",
        key: defaultKind.key ?? trackLabel,
    };
}

export function resolveReplaySettings(
    settings: UserSettings,
    experiment: Pick<Experiment, "logdir" | "trainer"> | null,
): ReplayRuleResolution {
    const trainerName =
        experiment != null && typeof experiment.trainer?.name === "string"
            ? experiment.trainer.name.trim()
            : "";
    const logdir = experiment?.logdir?.trim() ?? "";

    if (logdir.length > 0 && Object.hasOwn(settings.replay.experimentRules, logdir)) {
        return {
            onlySavedActions: settings.replay.experimentRules[logdir],
            source: "experiment",
            key: logdir,
        };
    }

    if (trainerName.length > 0 && Object.hasOwn(settings.replay.trainerRules, trainerName)) {
        return {
            onlySavedActions: settings.replay.trainerRules[trainerName],
            source: "trainer",
            key: trainerName,
        };
    }

    return {
        onlySavedActions: settings.replay.globalOnlySavedActions,
        source: "global",
        key: null,
    };
}
