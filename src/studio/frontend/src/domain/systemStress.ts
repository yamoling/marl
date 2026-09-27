/**
 * Device load helpers for the device picker (port of the old UI's `utils/systemStress.ts`),
 * with the product-spec warning threshold of 75 %. Inputs are percentages.
 */
export const STRESS_WARNING_THRESHOLD = 75;

export type Usage = { cpu: number; ram: number; gpus: { index: number; utilization: number; memory: number }[] };
export type DeviceOption = { value: string; label: string; stress: number };

export const gpuStress = (g: Usage["gpus"][number]) => Math.max(g.utilization, g.memory);
const cpuStress = (u: Usage) => Math.max(u.cpu, u.ram);

/** @ai-generated */
export function overallStress(u: Usage | null): number {
  if (!u) return 0;
  return Math.max(u.cpu, u.ram, ...u.gpus.map(gpuStress));
}

export function stressLabel(v: number): string {
  if (v < 40) return "Low";
  if (v < 60) return "Moderate";
  if (v < STRESS_WARNING_THRESHOLD) return "High";
  return "Critical";
}

export function stressColour(v: number): string {
  if (v < 40) return "var(--ok)";
  if (v < 60) return "#b8a100";
  if (v < STRESS_WARNING_THRESHOLD) return "var(--hot)";
  return "var(--err)";
}

/** Auto, CPU and one option per GPU, with their current load. @ai-generated */
export function deviceOptions(u: Usage | null): DeviceOption[] {
  if (!u) return [{ value: "auto", label: "Auto", stress: 0 }];
  return [
    { value: "auto", label: "Auto", stress: overallStress(u) },
    { value: "cpu", label: "CPU", stress: cpuStress(u) },
    ...u.gpus.map((g) => ({ value: `cuda:${g.index}`, label: `GPU ${g.index}`, stress: gpuStress(g) })),
  ];
}

/** Least loaded concrete device (CPU or a GPU). @ai-generated */
export function recommendedDevice(u: Usage | null): DeviceOption {
  const opts = deviceOptions(u).filter((o) => o.value !== "auto");
  if (!opts.length) return { value: "auto", label: "Auto", stress: 0 };
  return opts.reduce((best, o) => (o.stress < best.stress ? o : best));
}

/**
 * Load of what the run would use: the chosen device, or with "auto" the most loaded GPU still
 * enabled (else the overall load).
 *
 * @ai-generated
 */
export function selectedStress(u: Usage | null, device: string, disabled: number[] = []): number | null {
  if (!u) return null;
  if (device === "cpu") return cpuStress(u);
  if (device === "auto") {
    const gpus = u.gpus.filter((g) => !disabled.includes(g.index));
    return gpus.length ? Math.max(...gpus.map(gpuStress)) : cpuStress(u);
  }
  const g = u.gpus.find((x) => `cuda:${x.index}` === device || (device === "cuda" && x.index === 0));
  return g ? gpuStress(g) : null;
}

/** Warning when the selected device is above the threshold (with an alternative). @ai-generated */
export function deviceWarning(u: Usage | null, device: string, disabled: number[] = []): string | null {
  const s = selectedStress(u, device, disabled);
  if (s === null || s < STRESS_WARNING_THRESHOLD) return null;
  const rec = recommendedDevice(u);
  const alt = rec.value !== device && rec.stress < STRESS_WARNING_THRESHOLD ? ` Recommended alternative: ${rec.label} (${Math.round(rec.stress)}%).` : "";
  return `The selected device is at ${Math.round(s)}% load.${alt}`;
}
