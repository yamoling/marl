import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL, wsURL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";
import { useExperimentStore } from "./ExperimentStore";
import { Experiment } from "../models/Experiment";

export const useRunnerStore = defineStore("RunnerStore", () => {
    const loading = ref(false);
    const experimentStore = useExperimentStore();


    async function createRunner(experiment_logdir: string, checkpoint: string | null = null) {
        loading.value = true;
        try {
            const resp = await fetch(`${HTTP_URL}/runner/create/${experiment_logdir}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ checkpoint })
            });
            const runner = await resp.json();
            loading.value = false;
            return runner;
        } catch (e: any) {
            loading.value = false;
            throw e;
        }
    }

    async function startTraining(logdir: string, numSteps: number, testInterval: number, numTests: number): Promise<number> {
        const resp = await fetch(`${HTTP_URL}/runner/train/start/${logdir}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_steps: numSteps,
                test_interval: testInterval,
                num_tests: numTests
            })
        });
        const port = await resp.json() as { port: number };
        console.log(port)
        setTimeout(() => listen(logdir, port.port), 200);
        return port.port;
    }

    function listen(logdir: string, port: number) {
        const ws = new WebSocket(wsURL(port));
        const experiment = experimentStore.loadedExperiments.get(logdir) as Experiment;
        const trainList = experiment.train;
        const testList = experiment.test;
        ws.onmessage = (event: MessageEvent) => {
            console.log(event.data);
            const data = JSON.parse(event.data) as ReplayEpisodeSummary;
            // Remove the logdir from data.directory
            const directory = data.directory.replace(logdir, "");
            if (directory.startsWith("test")) {
                testList.push(data);
            } else {
                trainList.push(data);
            }
        }
    }

    return { startTraining };
});
