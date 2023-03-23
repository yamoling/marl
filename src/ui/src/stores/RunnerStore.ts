import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL, wsURL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";

export const useRunnerStore = defineStore("RunnerStore", () => {
    const loading = ref(false);
    const runners = new Map<string, string>();

    async function createRunner(logdir: string, checkpoint: string | null = null) {
        loading.value = true;
        const resp = await fetch(`${HTTP_URL}/runner/create/${logdir}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ checkpoint })
        });
        const { port } = await resp.json() as { port: number };
        runners.set(logdir, wsURL(port));
        loading.value = false;
    }

    async function registerObserver(logdir: string, notify: (data: ReplayEpisodeSummary | null) => void) {
        if (!runners.has(logdir)) {
            await createRunner(logdir);
        }
        const url = runners.get(logdir) as string;
        const ws = new WebSocket(url);
        ws.onmessage = (event: MessageEvent) => {
            if ((event.data as string).length == 0) {
                notify(null);
            } else {
                notify(JSON.parse(event.data) as ReplayEpisodeSummary | null);
            }
        };
        ws.onclose = () => {
            console.log("Connection lost at ", url);
        }
    }

    async function startTraining(
        logdir: string,
        numSteps: number,
        testInterval: number,
        numTests: number,
    ) {
        await fetch(`${HTTP_URL}/runner/train/start/${logdir}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_steps: numSteps,
                test_interval: testInterval,
                num_tests: numTests
            })
        });
    }
    return { startTraining, registerObserver };
});
