import { z } from "zod";

export const RewardSpaceSchema = z.object({
    size: z.number().optional(),
    labels: z.array(z.string()).default([]),
}).passthrough();

export const ActionSpaceSchema = z.object({
    space_type: z.union([z.literal("discrete"), z.literal("continuous")]).optional(),
    shape: z.array(z.number()).default([]),
    size: z.number().optional(),
    labels: z.array(z.string()).default([]),
    low: z.array(z.number()).nullable().optional(),
    high: z.array(z.number()).nullable().optional(),
    space_class: z.string().optional(),
}).passthrough();

const SerializableObjectSchema = z.object({
    "class-name": z.string(),
});

export const EnvConfigSchema = SerializableObjectSchema.extend({
    agent_id: z.boolean().default(true),
    time_limit: z.number().nullable().default(null),
    last_action: z.boolean().default(false),
});

export const LLEConfigSchema = EnvConfigSchema.extend({
    "class-name": z.literal("LLEConfig"),
    level_or_path: z.union([z.number(), z.string()]),
    obs_type: z.enum(["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]).default("layered"),
    state_type: z.enum(["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]).default("state"),
});

export const SMACConfigSchema = EnvConfigSchema.extend({
    "class-name": z.literal("SMACConfig"),
    map_name: z.string(),
    debug: z.boolean().default(false),
});

export const EnvSchema = z.union([LLEConfigSchema, SMACConfigSchema]);

export type EnvConfig = z.infer<typeof EnvConfigSchema>;
export type LLEConfig = z.infer<typeof LLEConfigSchema>;
export type SMACConfig = z.infer<typeof SMACConfigSchema>;
export type Env = z.infer<typeof EnvSchema>;
export type ActionSpace = z.infer<typeof ActionSpaceSchema>;
export type DiscreteActionSpace = ActionSpace;
export type RewardSpace = z.infer<typeof RewardSpaceSchema>;

export function getEnvDisplayName(env: Env): string {
    switch (env["class-name"]) {
        case "LLEConfig":
            return String(env.level_or_path);
        case "SMACConfig":
            return env.map_name;
        default:
            return env["class-name"];
    }
}