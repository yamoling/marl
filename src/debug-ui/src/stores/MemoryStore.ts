import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Transition } from "../models/Episode";

export const useMemoryStore = defineStore("MemoryStore", () => {

    async function getPriorities(): Promise<{ cumsum: number, priorities: number[] }> {
        const resp = await fetch(`${HTTP_URL}/train/memory/priorities`);
        return await resp.json();
    }

    async function getTransition(indexInMemory: number): Promise<Transition> {
        const resp = await fetch(`${HTTP_URL}/train/memory/transition/${indexInMemory}`);
        return await resp.json();
    }

    return { getPriorities, getTransition };
});
