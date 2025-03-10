import { ref } from "vue";
import { defineStore } from "pinia";
import { wsURL } from "../constants";
import { SystemInfo } from "../models/SystemInfo";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);


    function updateSystemInfo() {
        const address = `ws://${location.hostname}:5001`;
        console.log("Connecting to system info websocket", address);
        const ws = new WebSocket(address);
        ws.onopen = () => {
            console.log("Connected to system info websocket")
        }
        ws.onmessage = async (event) => {
            const blob = event.data as Blob;
            const text = await blob.text()
            systemInfo.value = JSON.parse(text) as SystemInfo;
        }
        ws.onerror = () => {
            console.error("Error connecting to system info websocket, retrying in 10 seconds")
            systemInfo.value = null;
            setTimeout(updateSystemInfo, 10_000);
        }
        ws.onclose = () => {
            console.warn("Lost system info websocket connection, retrying in 5 seconds")
            systemInfo.value = null;
            setTimeout(updateSystemInfo, 5000);
        }
    }
    updateSystemInfo();

    return { systemInfo };
});
