/**
 * System readings (CPU, RAM, GPUs) from `api.subscribeSystem`, lightly smoothed (EMA), and the
 * GPU list for the device picker.
 */
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useApi, type ConnectionState, type SystemReading } from "../api";

export type GpuUsage = { index: number; utilization: number; memory: number; usedMB: number; totalMB: number };
const ALPHA = 0.5;
export const HOT_THRESHOLD = 75;

export const useSystemStore = defineStore("system", () => {
  const state = ref<ConnectionState>("connecting");
  const cpu = ref(0);
  const ram = ref(0);
  const gpus = ref<GpuUsage[]>([]);
  const hasReading = ref(false);
  let stopFn: (() => void) | null = null;

  const ema = (prev: number, next: number) => (hasReading.value ? prev + ALPHA * (next - prev) : next);

  /** Merge one reading (percentages for CPU/RAM, ratios for GPUs). @ai-generated */
  function onReading(r: SystemReading): void {
    cpu.value = ema(cpu.value, r.cpu);
    ram.value = ema(ram.value, r.ram);
    gpus.value = r.gpus.map((g) => {
      const prev = gpus.value.find((x) => x.index === g.index);
      return {
        index: g.index,
        utilization: prev && hasReading.value ? ema(prev.utilization, g.utilization * 100) : g.utilization * 100,
        memory: prev && hasReading.value ? ema(prev.memory, g.memory_usage * 100) : g.memory_usage * 100,
        usedMB: g.used_memory,
        totalMB: g.total_memory,
      };
    });
    hasReading.value = true;
  }

  function start(): void {
    if (stopFn) return;
    stopFn = useApi().subscribeSystem(onReading, (s) => (state.value = s));
  }

  function stop(): void {
    stopFn?.();
    stopFn = null;
  }

  const rows = computed(() => [
    { key: "CPU", value: cpu.value, tip: `CPU ${Math.round(cpu.value)}%` },
    { key: "RAM", value: ram.value, tip: `RAM ${Math.round(ram.value)}%` },
    ...gpus.value.map((g) => ({
      key: `GPU${g.index}`,
      value: Math.max(g.utilization, g.memory),
      tip: `GPU ${g.index}: utilisation ${Math.round(g.utilization)}% · memory ${Math.round(g.usedMB)} / ${Math.round(g.totalMB)} MB (${Math.round(g.memory)}%)`,
    })),
  ]);

  return { state, cpu, ram, gpus, hasReading, rows, start, stop, onReading };
});
