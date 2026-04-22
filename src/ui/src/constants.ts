export const HOST = "localhost" as const;
export const HTTP_PORT = 5000 as const;
export const HTTP_URL = `http://${HOST}:${HTTP_PORT}` as const;
export function wsURL(port: number) {
  return `ws://${HOST}:${port}`;
}
export const OBS_TYPES = [
  "RGB_IMAGE",
  "FLATTENED",
  "LAYERED",
  "RELATIVE_POSITIONS",
  "FEATURES",
  "FEATURES2",
] as const;
export const POLICIES = [
  "SoftmaxPolicy",
  "EpsilonGreedy",
  "DecreasingEpsilonGreedyPolicy",
  "ArgMax",
] as const;

export const CATEGORY_COLOURS = [
  "#1f77b4", // blue
  "#ff7f0e", // orange
  "#2ca02c", // green
  "#d62728", // red
  "#9467bd", // purple
  "#8c564b", // brown
  "#e377c2", // pink
  "#7f7f7f", // gray
  "#bcbd22", // olive
  "#17becf", // cyan
  "#393b79", // indigo
  "#637939", // sage green
  "#8c6d31", // tan
  "#843c39", // maroon
  "#7b4173", // plum
  "#3182bd", // steel blue
] as const;
