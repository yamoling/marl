import { defineStore } from "pinia";
import { HTTP_URL, wsURL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";
import { RunConfig } from "../models/Runs";
import { useExperimentStore } from "./ExperimentStore";

export const useRunnerStore = defineStore("RunnerStore", () => {
    const webSockets = new Map<string, WebSocket>();
    const experimentStore = useExperimentStore();

    async function createRunner(runConfig: RunConfig) {
        const resp = await fetch(`${HTTP_URL}/runner/create/${runConfig.logdir}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(runConfig)
        });
        const ports = await resp.json() as [string, number][];
        return ports;
    }

    async function restartTraining(
        rundir: string,
        numSteps: number,
        testInterval: number,
        numTests: number,
    ) {
        await fetch(`${HTTP_URL}/runner/restart/${rundir}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_steps: numSteps,
                test_interval: testInterval,
                num_tests: numTests
            })
        });
    }

    async function startListening(
        rundir: string,
        port: number,
        onTrainUpdate: (data: ReplayEpisodeSummary) => void | null,
        onTestUpdate: (data: ReplayEpisodeSummary) => void,
        onCloseFunction: () => void,
        attempt: number = 0
    ) {
        if (attempt > 5) {
            throw new Error("Failed to connect to websocket");
        }
        // Close the previous connection if it exists
        stopListening(rundir);
        const url = wsURL(port);

        const ws = new WebSocket(url);
        ws.onerror = () => setTimeout(() => startListening(rundir, port, onTrainUpdate, onTestUpdate, onCloseFunction, attempt + 1), 1000);

        const rundirLength = rundir.length;
        ws.onopen = () => webSockets.set(rundir, ws);

        ws.onmessage = (event: MessageEvent) => {
            const data = JSON.parse(event.data) as ReplayEpisodeSummary;
            const directory = data.directory.slice(rundirLength);
            if (directory.startsWith("/train") && onTrainUpdate != null) {
                onTrainUpdate(data);
            } else {
                onTestUpdate(data)
            }
        };
        ws.onclose = () => {
            stopListening(rundir);
            onCloseFunction();
        };
        return;
    }

    function stopListening(rundir: string) {
        webSockets.get(rundir)?.close();
    }

    async function deleteRun(rundir: string) {
        return fetch(`${HTTP_URL}/runner/delete/${rundir}`, {
            method: "DELETE"
        });
    }

    async function stopRunner(rundir: string) {
        return fetch(`${HTTP_URL}/runner/stop/${rundir}`, {
            method: "POST"
        });
    }

    async function getRunnerPort(rundir: string): Promise<number | null> {
        try {
            const resp = await fetch(`${HTTP_URL}/runner/port/${rundir}`);
            // Check response status
            if (resp.status !== 200) {
                return null;
            }
            return Number.parseInt(await resp.text());
        }
        catch (e) {
            return null;
        }
    }

    return { startListening, stopListening, createRunner, deleteRun, stopRunner, restartTraining, getRunnerPort };
});
