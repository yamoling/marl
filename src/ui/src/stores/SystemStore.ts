import { ref, computed } from "vue";
import { defineStore } from "pinia";
import { SystemInfo, SystemReadingSchema, SystemUsage } from "../models/SystemInfo";
import { HTTP_URL } from "../constants";

const STRESS_LEVELS = [
    { value: 0, label: "Low", color: "rgb(34, 197, 94)" },
    { value: 40, label: "Moderate", color: "rgb(234, 179, 8)" },
    { value: 70, label: "High", color: "rgb(249, 115, 22)" },
    { value: 85, label: "Critical", color: "rgb(239, 68, 68)" },
]


export const useSystemStore = defineStore("SystemStore", () => {
    const systemSmoother = new SystemInfo();
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let addressIndex = 0;
    const systemUsage = ref<SystemUsage | null>(null);
    const cpu = computed(() => systemUsage.value?.cpu ?? 0);
    const ram = computed(() => systemUsage.value?.ram ?? 0);
    const gpus = computed(() => systemUsage.value?.gpus ?? []);
    const globalGpuUsage = computed(() => {
        const meanUtilization = systemUsage.value?.gpus.reduce((sum, gpu) => sum + gpu.utilization, 0) ?? 0;
        const meanMemory = systemUsage.value?.gpus.reduce((sum, gpu) => sum + gpu.memory, 0) ?? 0;
        return Math.max(meanUtilization, meanMemory);
    });

    function getAddresses() {
        const protocol = location.protocol === "https:" ? "wss" : "ws";
        const hosts = [location.host, `${location.hostname}:5000`];
        const uniqueHosts = [...new Set(hosts)];
        return uniqueHosts.map((host) => `${protocol}://${host}/ws/system-info`);
    }

    function scheduleReconnect(delayMs: number) {
        if (reconnectTimeout != null) {
            return;
        }
        reconnectTimeout = setTimeout(() => {
            reconnectTimeout = null;
            updateSystemInfoLoop();
        }, delayMs);
    }

    function updateSystemInfoLoop() {
        if (ws != null && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
            return;
        }

        const addresses = getAddresses();
        const safeIndex = Math.min(addressIndex, addresses.length - 1);
        const address = addresses[safeIndex];
        const nextWs = new WebSocket(address);
        ws = nextWs;

        nextWs.onopen = () => {
            addressIndex = 0;
        }

        nextWs.onmessage = async (event) => {
            const blob = event.data as Blob;
            const text = await blob.text();
            const data = JSON.parse(text);
            const reading = SystemReadingSchema.parse(data);
            const res = systemSmoother.addReading(reading)
            systemUsage.value = res;
        }

        nextWs.onerror = () => {
            console.info("Error on system info websocket")
        }

        nextWs.onclose = () => {
            if (ws !== nextWs) {
                return;
            }
            ws = null;
            if (safeIndex < addresses.length - 1) {
                addressIndex = safeIndex + 1;
                console.warn(`Lost system info websocket connection to ${address}, trying fallback endpoint`)
                scheduleReconnect(0);
                return;
            }

            addressIndex = 0;
            console.warn("Lost system info websocket connection, retrying in 5 seconds")
            scheduleReconnect(5000);
        }
    }

    async function getSystemSpecs() {
        const resp = await fetch(`${HTTP_URL}/system-specs`);
        const data = await resp.json();
        const reading = SystemReadingSchema.parse(data);
        systemUsage.value = systemSmoother.addReading(reading);
    }
    // Fetch with http the first time because it is faster than establishing the websocket connection.
    getSystemSpecs();
    // Then update with websocket data and keep it updated.
    updateSystemInfoLoop()


    function stressLabelFor(usage: number): string {
        for (let i = STRESS_LEVELS.length - 1; i >= 0; i--) {
            if (usage >= STRESS_LEVELS[i].value) {
                return STRESS_LEVELS[i].label;
            }
        }
        return STRESS_LEVELS[0].label;
    }

    function stressColorFor(usage: number): string {
        for (let i = STRESS_LEVELS.length - 1; i >= 0; i--) {
            if (usage >= STRESS_LEVELS[i].value) {
                return STRESS_LEVELS[i].color;
            }
        }
        return STRESS_LEVELS[0].color;
    }
    return { systemUsage, cpu, ram, gpus, globalGpuUsage, stressLabelFor, stressColorFor };
});
