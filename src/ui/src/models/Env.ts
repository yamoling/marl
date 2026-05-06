import { z } from "zod";


export const SpaceSchema = z.object({
    shape: z.array(z.number()),
    size: z.number(),
    labels: z.array(z.string())
});

export const MultiDiscreteSpaceSchema = SpaceSchema.extend({
    spaces: z.array(SpaceSchema),
    n_dims: z.number(),
});

export const ContinuousSpaceSchema = SpaceSchema.extend({
    low: z.array(z.number()),
    high: z.array(z.number()),
});

export const ActionSpaceSchema = z.union([MultiDiscreteSpaceSchema, ContinuousSpaceSchema]);

export const EnvSchema = z.object({
    action_space: ActionSpaceSchema,
    observation_shape: z.array(z.number()),
    extras_shape: z.array(z.number()),
    n_agents: z.number(),
    state_shape: z.array(z.number()),
    extras_meanings: z.array(z.string()),
    reward_space: z.union([SpaceSchema, ContinuousSpaceSchema]),
}).loose();

export const EnvConfigSchema = z.object({
    name: z.string(),
    agent_id: z.boolean(),
    time_limit: z.number().optional(),
    last_action: z.boolean(),
    maven_noise_size: z.number().optional(),
    env: EnvSchema.optional(),
}).loose();




export const LLEConfigSchema = EnvConfigSchema.extend({
    level_or_path: z.union([z.number(), z.string()]),
    obs_type: z.enum(["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]),
    state_type: z.enum(["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]),
});

export const SMACConfigSchema = EnvConfigSchema.extend({
    map_name: z.string(),
    debug: z.boolean().default(false),
});

export type ActionSpace = z.infer<typeof MultiDiscreteSpaceSchema> | z.infer<typeof ContinuousSpaceSchema>;
export type EnvConfig = z.infer<typeof EnvConfigSchema>;
