import { z } from "zod";
import type { Experiment } from "./Experiment";
import type { TimelineTrackKind } from "./Timeline";

export const SETTINGS_SCHEMA_VERSION = 2 as const;

export type ReplayRuleSource = "global" | "trainer" | "experiment";
export type TrackRuleSource = "global" | "trainer" | "experiment" | "default";

export interface ReplaySettings {
  globalOnlySavedActions: boolean;
  trainerRules: Record<string, boolean>;
}

export interface TrackSettings {
  defaultKinds: Record<string, TimelineTrackKind>;
}

export interface VisualizationSettings {
  colours: Record<string, string>;
  tracks: TrackSettings;
  useWallTime: boolean;
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

export function matchesTrackRuleKey(
  ruleKey: string,
  trackLabel: string,
): boolean {
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

const ReplaySettingsSchema = z.object({
  globalOnlySavedActions: z.boolean(),
  trainerRules: ReplayRuleMapSchema,
});

const TrackSettingsSchema = z.object({
  defaultKinds: z.record(z.string(), TrackKindSchema),
});

const VisualizationSettingsSchema = z.object({
  colours: z.record(z.string(), z.string()),
  tracks: TrackSettingsSchema,
  useWallTime: z.boolean().default(false),
});

const UserSettingsSchema = z.object({
  version: z.literal(SETTINGS_SCHEMA_VERSION),
  replay: ReplaySettingsSchema,
  visualization: VisualizationSettingsSchema,
});

export function createDefaultSettings(): UserSettings {
  return {
    version: SETTINGS_SCHEMA_VERSION,
    replay: {
      globalOnlySavedActions: false,
      trainerRules: {},
    },
    visualization: {
      colours: {},
      tracks: {
        defaultKinds: {
          "/option.*/i": "categorical",
          "/reward.*/i": "numeric",
          "/{probability,probabilities}/i": "numeric",
        },
      },
      useWallTime: false,
    },
  };
}

export function parseSettings(raw: unknown): UserSettings | null {
  const parsed = UserSettingsSchema.safeParse(raw);
  return parsed.success ? parsed.data : null;
}

export function resolveTrackRule(
  settings: UserSettings,
  trackLabel: string,
  fallbackKind: TimelineTrackKind = "numeric",
): TrackRuleResolution {
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

  if (
    trainerName.length > 0 &&
    Object.hasOwn(settings.replay.trainerRules, trainerName)
  ) {
    return {
      onlySavedActions: settings.replay.trainerRules[trainerName],
      source: "trainer",
      key: trainerName,
    };
  }

  if (trainerName.length > 0) {
    let matchedKey: string | null = null;
    let matchedValue: boolean | null = null;
    for (const [ruleKey, value] of Object.entries(
      settings.replay.trainerRules,
    )) {
      if (!matchesTrackRuleKey(ruleKey, trainerName)) {
        continue;
      }
      if (matchedKey == null || ruleKey.length > matchedKey.length) {
        matchedKey = ruleKey;
        matchedValue = value;
      }
    }

    if (matchedKey != null && matchedValue != null) {
      return {
        onlySavedActions: matchedValue,
        source: "trainer",
        key: matchedKey,
      };
    }
  }

  return {
    onlySavedActions: settings.replay.globalOnlySavedActions,
    source: "global",
    key: null,
  };
}
