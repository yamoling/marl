export const HOST = "0.0.0.0" as const;
export const HTTP_PORT = 5000 as const;
export const HTTP_URL = `http://${HOST}:${HTTP_PORT}` as const;
export function wsURL(port: number) {
    return `ws://${HOST}:${port}`
}
export const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"] as const;
export const OBS_TYPES = ["RGB_IMAGE", "FLATTENED", "LAYERED", "RELATIVE_POSITIONS"] as const;
export const POLICIES = ["SoftmaxPolicy", "EpsilonGreedy", "DecreasingEpsilonGreedyPolicy", "ArgMax"] as const;

