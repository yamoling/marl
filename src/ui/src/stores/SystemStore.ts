import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { SystemInfo } from "../models/SystemInfo";

export const useSystemStore = defineStore("SystemStore", () => {
    const systemInfo = ref(null as SystemInfo | null);


    async function updateSystemInfo() {
        try {
            const response = await fetch(`${HTTP_URL}/system/usage`);
            systemInfo.value = await response.json();
            setTimeout(updateSystemInfo, 2000);
        } catch (e) {
            systemInfo.value = null;
            setTimeout(updateSystemInfo, 10_000);
        }

    }
    updateSystemInfo();

    return { systemInfo };
});
