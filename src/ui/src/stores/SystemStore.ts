import { ref } from "vue";
import { defineStore } from "pinia";
import { SystemInfo, SystemInfoSchema } from "../models/SystemInfo";
import { fromJsonString } from "../utils";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let addressIndex = 0;

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
            updateSystemInfo();
        }, delayMs);
    }

    function updateSystemInfo() {
        if (ws != null && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
            return;
        }

        const addresses = getAddresses();
        const safeIndex = Math.min(addressIndex, addresses.length - 1);
        const address = addresses[safeIndex];
        console.info("Connecting to system info websocket", address);
        const nextWs = new WebSocket(address);
        ws = nextWs;

        nextWs.onopen = () => {
            addressIndex = 0;
            console.info("Connected to system info websocket")
        }

        nextWs.onmessage = async (event) => {
            const blob = event.data as Blob;
            const text = await blob.text();
            systemInfo.value = fromJsonString(text, SystemInfoSchema, null);
        }

        nextWs.onerror = () => {
            console.error("Error on system info websocket")
        }

        nextWs.onclose = () => {
            if (ws !== nextWs) {
                return;
            }

            ws = null;
            systemInfo.value = null;

            if (safeIndex < addresses.length - 1) {
                addressIndex = safeIndex + 1;
                console.warn("Lost system info websocket connection, trying fallback endpoint")
                scheduleReconnect(0);
                return;
            }

            addressIndex = 0;
            console.warn("Lost system info websocket connection, retrying in 5 seconds")
            scheduleReconnect(5000);
        }
    }

    updateSystemInfo();

    return { systemInfo };
});
