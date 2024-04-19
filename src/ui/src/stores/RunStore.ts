import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import {Run} from "../models/Run";
import { ref } from "vue";

export const useRunStore = defineStore("RunStore", () => {
    const runs = ref(new Map<string, Run[]>());

    async function refresh(logdir: string) {
        const updatedRuns = await getRuns(logdir);
        runs.value.set(logdir, updatedRuns);
    }

    async function getRuns(logdir: string): Promise<Run[]> {
        const resp = await fetch(`${HTTP_URL}/runs/get/${logdir}`);
        return await resp.json();
    }

    async function remove(logdir: string) {
        runs.value.delete(logdir);
    }

    return {
        runs,
        getRuns,
        refresh,
        remove
    };
});
