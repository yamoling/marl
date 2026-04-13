import { GpuInfo, SystemInfo } from "../models/SystemInfo";

export const STRESS_WARNING_THRESHOLD = 85;
export const STRESS_SMOOTHING_WINDOW_MS = 10_000; // 3 second buffer for smoothing

/**
 * Temporal stress filter that smooths out rapid oscillations by maintaining a sliding window
 * of recent readings and returning a weighted average. This prevents the stress indicator
 * from flickering when values oscillate near threshold boundaries.
 */
export class TemporalStressFilter {
    private readings: Array<{ value: number; timestamp: number }> = [];
    private windowMs: number;

    constructor(windowMs: number = STRESS_SMOOTHING_WINDOW_MS) {
        this.windowMs = windowMs;
    }

    /**
     * Add a new stress reading and return the smoothed stress value.
     */
    addReading(value: number): number {
        const now = Date.now();
        this.readings.push({ value, timestamp: now });

        // Remove readings outside the window
        this.readings = this.readings.filter((r) => now - r.timestamp <= this.windowMs);

        // Return weighted average: recent readings have higher weight
        if (this.readings.length === 0) {
            return value;
        }
        if (this.readings.length === 1) {
            return this.readings[0].value;
        }

        const oldest = this.readings[0].timestamp;
        const newest = this.readings[this.readings.length - 1].timestamp;
        const range = Math.max(newest - oldest, 1); // Avoid division by zero

        let totalWeight = 0;
        let weightedSum = 0;

        for (const reading of this.readings) {
            // Weight increases linearly over time (newer = heavier)
            const age = reading.timestamp - oldest;
            const weight = 1 + age / range;
            weightedSum += reading.value * weight;
            totalWeight += weight;
        }

        return weightedSum / totalWeight;
    }

    /**
     * Reset the filter (clear all readings).
     */
    reset(): void {
        this.readings = [];
    }
}

export interface DeviceOption {
    value: string;
    label: string;
    stress: number;
}

export function getCpuUsage(systemInfo: SystemInfo): number {
    if (systemInfo.cpus.length === 0) {
        return 0;
    }
    const total = systemInfo.cpus.reduce((acc, usage) => acc + usage, 0);
    return total / systemInfo.cpus.length;
}

export function getGpuStress(gpu: GpuInfo): number {
    return Math.max(gpu.utilization, gpu.memory_usage) * 100;
}

export function getGpuAggregateStress(systemInfo: SystemInfo): number {
    if (systemInfo.gpus.length === 0) {
        return 0;
    }
    return Math.max(...systemInfo.gpus.map(getGpuStress));
}

export function getOverallStress(systemInfo: SystemInfo): number {
    return Math.max(getCpuUsage(systemInfo), systemInfo.ram, getGpuAggregateStress(systemInfo));
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

export function buildDeviceOptions(systemInfo: SystemInfo | null): DeviceOption[] {
    if (systemInfo == null) {
        return [{ value: "auto", label: "Auto", stress: 0 }];
    }

    const options: DeviceOption[] = [
        { value: "auto", label: "Auto", stress: getOverallStress(systemInfo) },
        { value: "cpu", label: "CPU", stress: Math.max(getCpuUsage(systemInfo), systemInfo.ram) },
    ];

    for (const gpu of systemInfo.gpus) {
        options.push({
            value: `cuda:${gpu.index}`,
            label: `GPU ${gpu.index}`,
            stress: getGpuStress(gpu),
        });
    }

    return options;
}

export function buildGpuDeviceOptions(systemInfo: SystemInfo | null): DeviceOption[] {
    return buildDeviceOptions(systemInfo).filter(option => option.value.startsWith("cuda:"));
}

export function getDefaultSelectedGpuDevices(systemInfo: SystemInfo | null): string[] {
    if (systemInfo == null) {
        return [];
    }

    return systemInfo.gpus
        .filter(gpu => getStressLabel(getGpuStress(gpu)) !== "High" && getStressLabel(getGpuStress(gpu)) !== "Critical")
        .map(gpu => `cuda:${gpu.index}`);
}

export function getDisabledDevicesFromSelected(systemInfo: SystemInfo | null, selectedDevices: string[]): number[] {
    if (systemInfo == null) {
        return [];
    }

    const selected = new Set(selectedDevices);
    return systemInfo.gpus
        .filter(gpu => !selected.has(`cuda:${gpu.index}`))
        .map(gpu => gpu.index);
}

export function getDeviceStress(systemInfo: SystemInfo | null, device: string): number | null {
    if (systemInfo == null) {
        return null;
    }

    if (device === "auto") {
        return getOverallStress(systemInfo);
    }
    if (device === "cpu") {
        return Math.max(getCpuUsage(systemInfo), systemInfo.ram);
    }

    const index = Number.parseInt(device.replace("cuda:", ""), 10);
    if (Number.isNaN(index)) {
        return null;
    }

    const gpu = systemInfo.gpus.find(item => item.index === index);
    if (gpu == null) {
        return null;
    }

    return getGpuStress(gpu);
}

export function getRecommendedDevice(systemInfo: SystemInfo | null): DeviceOption {
    const options = buildDeviceOptions(systemInfo).filter(option => option.value !== "auto");
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