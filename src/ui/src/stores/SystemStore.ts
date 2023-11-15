import { ref } from "vue";
import { defineStore } from "pinia";
import { wsURL } from "../constants";
import { SystemInfo } from "../models/SystemInfo";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);


    function updateSystemInfo() {
        const ws = new WebSocket(wsURL(5001));
        ws.onmessage = (event) => {
            systemInfo.value = JSON.parse(event.data) as SystemInfo;
        }
        ws.onclose = () => {
            console.log("Lost system info websocket connection, retrying in 10 seconds")
            systemInfo.value = null;
            setTimeout(updateSystemInfo, 10_000);
        }
    }
    updateSystemInfo();

    return { systemInfo };
});
