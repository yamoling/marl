import { z } from "zod";

export const RewardSpaceSchema = z.object({
    size: z.number(),
    labels: z.array(z.string()),
});

export const ActionSpaceSchema = z.object({
    space_type: z.union([z.literal("discrete").optional(), z.literal("continuous")]),
    shape: z.array(z.number()),
    size: z.number().optional(),
    labels: z.array(z.string()),
    low: z.array(z.number()).nullable().optional(),
    high: z.array(z.number()).nullable().optional(),
    space_class: z.string().optional(),
});


const envBase = {
    name: z.string(),
    n_agents: z.number(),
    n_actions: z.number(),
    observation_shape: z.array(z.number()),
    state_shape: z.array(z.number()),
    extras_shape: z.array(z.number()),
    extras_meanings: z.array(z.string()),
    action_space: ActionSpaceSchema,
    reward_space: RewardSpaceSchema,
}

export const EnvSchema = z.object(envBase);

export const EnvWrapperSchema = z.object(envBase).extend({
    full_name: z.string(),
    wrapped: z.union([EnvSchema, z.any()]),
});



export type Env = z.infer<typeof EnvSchema>;
export interface EnvWrapper extends Env {
    full_name: string,
    wrapped: Env | EnvWrapper
}
export type ActionSpace = z.infer<typeof ActionSpaceSchema>;
export type RewardSpace = z.infer<typeof RewardSpaceSchema>;



export function getWrapperParameters(env: EnvWrapper) {
    const { wrapped, full_name, ...parameters } = env;
    const childKeys = Object.keys(env.wrapped);
    return Object.entries(parameters)
        .filter(([key]) => !childKeys.includes(key))
        .reduce((result, [key, value]) => result.set(key, value), new Map<string, any>());
}