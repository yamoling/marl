/**
 * Episodes sheet state: the selected experiment and test step, the performance-timeline metric
 * (remembered per experiment in the workspace), the episodes at that step, the selected episode's
 * replay and its playback (port of the old `ReplayStore` + the `EpisodeReplay` logic).
 *
 * Entry point for other components: `openEpisodes(experimentId, step)` (step null = last test step).
 */
import { defineStore } from "pinia";
import { computed, markRaw, ref, shallowRef, watch } from "vue";
import { ApiError, isAbortError, useApi, type EpisodeSummary, type Issue, type ReplayEpisode } from "../api";
import { maxTime, REPLAY_BLOCKING_CODES } from "../domain/replay";
import { lastStep, metricKey, neighbourStep, snapStep, timelineMetric } from "../domain/timeline";
import type { MetricRef } from "../api/schemas";
import { useExperimentsStore } from "./experiments";
import { useSettingsStore } from "./settings";
import { useWorkspaceStore } from "./workspace";

type Status = "idle" | "loading" | "ok" | "error";
export type EpisodeRef = { run: string; test: number; step: number };

export const PLAYBACK_SPEEDS = [1, 2, 4, 8, 16] as const;
const EPISODES_DEBOUNCE_MS = 120;

export const useReplayStore = defineStore("replay", () => {
  const experiments = useExperimentsStore();
  const workspace = useWorkspaceStore();
  const settings = useSettingsStore();

  const open = ref(false);
  const experiment = ref<string | null>(null);
  const step = ref<number | null>(null);
  /** Test steps per experiment (cache). */
  const testSteps = ref<Record<string, number[]>>({});
  const stepsStatus = ref<Status>("idle");

  const episodes = shallowRef<EpisodeSummary[]>([]);
  const badEpisodes = ref(0);
  const episodesStatus = ref<Status>("idle");
  const episodesError = ref<string | null>(null);

  const selected = ref<EpisodeRef | null>(null);
  const replay = shallowRef<ReplayEpisode | null>(null);
  const replayStatus = ref<Status>("idle");
  /** Last replay failure; `status` 409 means the backend refused to replay the experiment. */
  const replayError = ref<{ message: string; issue: Issue | null; status: number | null } | null>(null);

  /** Playback: current time index in the replay, playing flag and speed (frames per second). */
  const t = ref(0);
  const playing = ref(false);
  const fps = ref<number>(4);

  const steps = computed(() => (experiment.value ? (testSteps.value[experiment.value] ?? []) : []));
  const detail = computed(() => (experiment.value ? experiments.detail(experiment.value) : null));
  const catalog = computed(() => (experiment.value ? experiments.catalog(experiment.value) : null));
  const metric = computed<MetricRef | null>(() =>
    experiment.value ? timelineMetric(catalog.value, workspace.ws.replayMetric[experiment.value]) : null,
  );
  /** `capabilities.replay`: true, false, or null while unknown (health check pending). */
  const canReplay = computed<boolean | null>(() => detail.value?.capabilities.replay ?? null);
  /** Issue explaining why replay is unavailable (from the experiment's issues or the 409 answer). @ai-generated */
  const blockingIssue = computed<Issue | null>(() => {
    const issues = detail.value?.issues ?? [];
    for (const code of REPLAY_BLOCKING_CODES) {
      const hit = issues.find((i) => i.code === code);
      if (hit) return hit;
    }
    return replayError.value?.issue ?? issues.find((i) => i.level === "error") ?? null;
  });
  const replayRule = computed(() => settings.replayFor(detail.value?.raw));
  const maxT = computed(() => (replay.value ? maxTime(replay.value) : 0));

  let seq = 0;
  let episodesAbort: AbortController | null = null;
  let replayAbort: AbortController | null = null;
  let episodesTimer: ReturnType<typeof setTimeout> | null = null;
  let playTimer: ReturnType<typeof setInterval> | null = null;
  const healthChecked = new Set<string>();

  /**
   * Open the sheet on `experimentId` at the test step nearest to `requested` (null = last test
   * step). The experiment does not need to be loaded in the workspace.
   *
   * @ai-generated
   */
  function openEpisodes(experimentId: string, requested: number | null): void {
    open.value = true;
    if (experimentId !== experiment.value) {
      experiment.value = experimentId;
      episodes.value = [];
      episodesStatus.value = "idle";
    }
    clearSelection();
    step.value = requested;
    void sync(experimentId, requested);
  }

  /** Fetch what the experiment needs (detail, capabilities, test steps), then the episodes. @ai-generated */
  async function sync(id: string, requested: number | null): Promise<void> {
    const my = ++seq;
    const entry = experiments.entry(id);
    if (!entry || (entry.status !== "ready" && entry.status !== "loading")) await experiments.fetch(id, { silent: true });
    if (my !== seq) return;
    void ensureCapabilities(id);
    const known = await loadSteps(id);
    if (my !== seq) return;
    step.value = requested === null ? lastStep(known) : known.length ? snapStep(known, requested) : requested;
    scheduleEpisodes(0);
  }

  /** Run the lazy health check once per experiment when `capabilities.replay` is unknown. @ai-generated */
  async function ensureCapabilities(id: string): Promise<void> {
    const d = experiments.detail(id);
    if (!d || d.capabilities.replay !== null || healthChecked.has(id)) return;
    healthChecked.add(id);
    try {
      await experiments.checkHealth(id);
    } catch {
      /* the replay request itself will report the problem */
    }
  }

  /** @ai-generated */
  async function loadSteps(id: string): Promise<number[]> {
    stepsStatus.value = "loading";
    try {
      const s = await useApi().getTestSteps(id);
      testSteps.value = { ...testSteps.value, [id]: s };
      stepsStatus.value = "ok";
      return s;
    } catch {
      stepsStatus.value = "error";
      return testSteps.value[id] ?? [];
    }
  }

  function scheduleEpisodes(delay = EPISODES_DEBOUNCE_MS): void {
    if (episodesTimer) clearTimeout(episodesTimer);
    episodesTimer = setTimeout(() => void loadEpisodes(), delay);
  }

  /** Fetch the episodes of the current experiment at the current step (latest request wins). @ai-generated */
  async function loadEpisodes(): Promise<void> {
    episodesTimer = null;
    const id = experiment.value;
    const s = step.value;
    episodesAbort?.abort();
    if (!id || s === null) {
      episodes.value = [];
      episodesStatus.value = s === null && stepsStatus.value !== "loading" ? "ok" : "idle";
      return;
    }
    const ctl = (episodesAbort = new AbortController());
    episodesStatus.value = "loading";
    episodesError.value = null;
    try {
      const r = await useApi().getEpisodes(id, s, ctl.signal);
      if (ctl.signal.aborted) return;
      episodes.value = markRaw(r.items);
      badEpisodes.value = r.bad.length;
      episodesStatus.value = "ok";
    } catch (e) {
      if (isAbortError(e) || ctl.signal.aborted) return;
      episodes.value = [];
      episodesStatus.value = "error";
      episodesError.value = (e as Error)?.message ?? String(e);
    }
  }

  /** Switch experiment (from the sheet's selector): opens at its last test step. */
  function setExperiment(id: string): void {
    if (id !== experiment.value) openEpisodes(id, null);
  }

  /** Move to the test step nearest to `x` (timeline click/drag); episodes are refetched (debounced). @ai-generated */
  function setStep(x: number): void {
    const s = steps.value.length ? snapStep(steps.value, x) : Math.max(0, Math.round(x));
    if (s === step.value) return;
    step.value = s;
    clearSelection();
    scheduleEpisodes();
  }

  /** Previous (-1) / next (+1) test step. */
  function stepBy(delta: number): void {
    if (step.value === null) return;
    const s = neighbourStep(steps.value, step.value, delta);
    if (s !== null) setStep(s);
  }

  /** Remember the timeline metric of the current experiment. */
  function setMetric(m: MetricRef): void {
    if (experiment.value) workspace.ws.replayMetric = { ...workspace.ws.replayMetric, [experiment.value]: metricKey(m) };
  }

  const isSelected = (e: EpisodeSummary): boolean =>
    !!selected.value && selected.value.run === e.run && selected.value.test === e.test && selected.value.step === e.step;

  /**
   * Select an episode and load its replay, unless the experiment is known not to be replayable
   * (the viewer then shows the blocking issue instead).
   *
   * @ai-generated
   */
  async function selectEpisode(e: EpisodeSummary): Promise<void> {
    clearSelection();
    selected.value = { run: e.run, test: e.test, step: e.step };
    if (canReplay.value === false) return;
    await loadReplay();
  }

  /** (Re)load the replay of the selected episode. @ai-generated */
  async function loadReplay(): Promise<void> {
    const sel = selected.value;
    if (!sel) return;
    replayAbort?.abort();
    const ctl = (replayAbort = new AbortController());
    replayStatus.value = "loading";
    replayError.value = null;
    try {
      const r = await useApi().getReplay(
        sel.run,
        { step: sel.step, test: sel.test, onlySavedActions: replayRule.value.onlySavedActions },
        ctl.signal,
      );
      if (ctl.signal.aborted) return;
      replay.value = markRaw(r);
      replayStatus.value = "ok";
      t.value = 0;
    } catch (err) {
      if (isAbortError(err) || ctl.signal.aborted) return;
      replayStatus.value = "error";
      const issue = err instanceof ApiError ? (err.issue ?? null) : null;
      replayError.value = { message: (err as Error)?.message ?? String(err), issue, status: err instanceof ApiError ? err.status : null };
      if (err instanceof ApiError && err.status === 409 && experiment.value)
        void experiments.checkHealth(experiment.value).catch(() => undefined);
    }
  }

  function clearSelection(): void {
    pause();
    replayAbort?.abort();
    selected.value = null;
    replay.value = null;
    replayStatus.value = "idle";
    replayError.value = null;
    t.value = 0;
  }

  function close(): void {
    open.value = false;
    pause();
  }

  // ---------------------------------------------------------------- playback

  function seek(x: number): void {
    t.value = Math.max(0, Math.min(maxT.value, Math.round(x)));
  }

  function pause(): void {
    if (playTimer) clearInterval(playTimer);
    playTimer = null;
    playing.value = false;
  }

  /** Play from the current frame (from the start when at the end), stopping on the last frame. @ai-generated */
  function play(): void {
    if (!replay.value) return;
    if (t.value >= maxT.value) t.value = 0;
    pause();
    playing.value = true;
    playTimer = setInterval(() => {
      if (t.value >= maxT.value) pause();
      else t.value++;
    }, 1000 / fps.value);
  }

  const togglePlay = () => (playing.value ? pause() : play());

  watch(fps, () => playing.value && play());

  // Test steps grow while runs are running: refresh them when the experiment's detail changes.
  watch(detail, (d, prev) => {
    if (open.value && d && prev && d !== prev && experiment.value) void loadSteps(experiment.value);
  });

  return {
    open,
    experiment,
    step,
    steps,
    stepsStatus,
    episodes,
    badEpisodes,
    episodesStatus,
    episodesError,
    selected,
    replay,
    replayStatus,
    replayError,
    t,
    playing,
    fps,
    maxT,
    detail,
    catalog,
    metric,
    canReplay,
    blockingIssue,
    replayRule,
    openEpisodes,
    setExperiment,
    setStep,
    stepBy,
    setMetric,
    isSelected,
    selectEpisode,
    loadReplay,
    clearSelection,
    close,
    seek,
    play,
    pause,
    togglePlay,
  };
});
