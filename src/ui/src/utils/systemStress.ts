import { GpuUsage, SystemUsage } from "../models/SystemInfo";

export const STRESS_WARNING_THRESHOLD = 85;

export interface DeviceOption {
  value: string;
  label: string;
  stress: number;
}

export function getCpuUsage(systemUsage: SystemUsage | null): number {
  return systemUsage?.cpu ?? 0;
}

export function getGpuStress(gpu: GpuUsage): number {
  return Math.max(gpu.utilization, gpu.memory);
}

export function getGpuAggregateStress(systemUsage: SystemUsage | null): number {
  if (systemUsage == null || systemUsage.gpus.length === 0) {
    return 0;
  }
  return Math.max(...systemUsage.gpus.map(getGpuStress));
}

export function getOverallStress(systemUsage: SystemUsage | null): number {
  return Math.max(
    getCpuUsage(systemUsage),
    systemUsage?.ram ?? 0,
    getGpuAggregateStress(systemUsage),
  );
}

export function getStressLabel(usage: number): string {
  if (usage < 40) return "Low";
  if (usage < 70) return "Moderate";
  if (usage < STRESS_WARNING_THRESHOLD) return "High";
  return "Critical";
}

export function getStressColor(usage: number): string {
  if (usage < 40) return "rgb(34, 197, 94)";
  if (usage < 70) return "rgb(234, 179, 8)";
  if (usage < STRESS_WARNING_THRESHOLD) return "rgb(249, 115, 22)";
  return "rgb(239, 68, 68)";
}

export function buildDeviceOptions(
  systemUsage: SystemUsage | null,
): DeviceOption[] {
  if (systemUsage == null) {
    return [{ value: "auto", label: "Auto", stress: 0 }];
  }

  const options: DeviceOption[] = [
    { value: "auto", label: "Auto", stress: getOverallStress(systemUsage) },
    {
      value: "cpu",
      label: "CPU",
      stress: Math.max(getCpuUsage(systemUsage), systemUsage.ram),
    },
  ];

  for (const gpu of systemUsage.gpus) {
    options.push({
      value: `cuda:${gpu.index}`,
      label: `GPU ${gpu.index}`,
      stress: getGpuStress(gpu),
    });
  }

  return options;
}

export function buildGpuDeviceOptions(
  systemUsage: SystemUsage | null,
): DeviceOption[] {
  return buildDeviceOptions(systemUsage).filter((option) =>
    option.value.startsWith("cuda:"),
  );
}

export function getDefaultSelectedGpuDevices(
  systemUsage: SystemUsage | null,
): string[] {
  if (systemUsage == null) {
    return [];
  }

  const recommendedDevice = getRecommendedDevice(systemUsage);
  return recommendedDevice.value.startsWith("cuda:") ? [recommendedDevice.value] : [];
}

export function getDisabledDevicesFromSelected(
  systemUsage: SystemUsage | null,
  selectedDevices: string[],
): number[] {
  if (systemUsage == null) {
    return [];
  }

  const selected = new Set(selectedDevices);
  return systemUsage.gpus
    .filter((gpu) => !selected.has(`cuda:${gpu.index}`))
    .map((gpu) => gpu.index);
}

export function getDeviceStress(
  systemUsage: SystemUsage | null,
  device: string,
): number | null {
  if (systemUsage == null) {
    return null;
  }

  if (device === "auto") {
    return getOverallStress(systemUsage);
  }
  if (device === "cpu") {
    return Math.max(getCpuUsage(systemUsage), systemUsage.ram);
  }

  const index = Number.parseInt(device.replace("cuda:", ""), 10);
  if (Number.isNaN(index)) {
    return null;
  }

  const gpu = systemUsage.gpus.find((item) => item.index === index);
  if (gpu == null) {
    return null;
  }

  return getGpuStress(gpu);
}

export function getRecommendedDevice(
  systemUsage: SystemUsage | null,
): DeviceOption {
  const options = buildDeviceOptions(systemUsage).filter(
    (option) => option.value !== "auto",
  );
  if (options.length === 0) {
    return { value: "auto", label: "Auto", stress: 0 };
  }

  return options.reduce((best, current) => {
    if (current.stress < best.stress) {
      return current;
    }
    return best;
  });
}
