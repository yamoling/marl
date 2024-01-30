import { ref } from "vue";
import { defineStore } from "pinia";
import { wsURL } from "../constants";
import { SystemInfo } from "../models/SystemInfo";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);


    function updateSystemInfo() {
        console.log("Connecting to system info websocket")
        const ws = new WebSocket("ws://0.0.0.0:8765");
        ws.onopen = () => {
            console.log("Connected to system info websocket")
            ws.send("coucou")
        }
        ws.onmessage = (event) => {
            systemInfo.value = JSON.parse(event.data) as SystemInfo;
            console.log("Received system info update", systemInfo.value);
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
