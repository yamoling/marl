import { ref } from "vue";
import { defineStore } from "pinia";
import { wsURL } from "../constants";
import { SystemInfo } from "../models/SystemInfo";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);


    function updateSystemInfo() {
        const ws = new WebSocket(wsURL(5001));
        ws.onopen = () => {
            console.info("Connected to system info websocket");
        }
        ws.onerror = (event) => {
            console.error("Error in system info websocket", event);
        }
        ws.onmessage = (event) => {
            systemInfo.value = JSON.parse(event.data) as SystemInfo;
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
