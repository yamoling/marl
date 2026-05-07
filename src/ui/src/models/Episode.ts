import { ActionSpace, ActionSpaceSchema } from "./Env";
import { Track, TrackGroup } from "./Timeline";
import { z } from "zod";

export const ReplayEpisodeSummarySchema = z.object({
  rundir: z.string(),
  time_step: z.number(),
  test_num: z.number(),
  metrics: z.record(z.string(), z.number()),
});
export type ReplayEpisodeSummary = z.infer<typeof ReplayEpisodeSummarySchema>;

/**
 * Step-wise agent details can be:
 *  - 0D directly a datapoint
 *  - 1D agent-wise scalar (agent-wise state-value estimation, agent-wise option selected, ...);
 *  - 2D agent-wise and <extra dimension>-wise (e.g. q-values, action probabilities, ...).
 */
export type AgentDetails = Record<string, number | number[] | number[][]>;

const ActionValueSchema = z.union([z.number(), z.array(z.number())]);
const AvailableActionSchema = z
  .union([z.boolean(), z.number()])
  .transform((value) => (typeof value === "number" ? value !== 0 : value));

export const EpisodeSchema = z.object({
  all_available_actions: z.array(z.array(z.array(AvailableActionSchema))),
  all_extras: z.array(z.array(z.array(z.number()))),
  all_observations: z.array(z.array(z.any())),
  actions: z.array(z.array(ActionValueSchema)),
  episode_len: z.number(),
  is_finished: z.boolean(),
  rewards: z.array(z.number()),
  states: z.array(z.array(z.array(z.number()))),
});

export const ReplayEpisodePayloadSchema = z.object({
  name: z.string(),
  directory: z.string().optional(),
  rundir: z.string().optional(),
  episode: EpisodeSchema,
  metrics: z.record(z.string(), z.number()),
  frames: z.array(z.string()),
  agent_details: z.array(z.record(z.string(), z.union([z.number(), z.array(z.number()), z.array(z.array(z.number()))]))),
  action_space: ActionSpaceSchema,
  replay_mismatch: z.boolean().default(false),
  mismatch_details: z.array(z.string()).default([]),
  replay_kind: z.enum(["CombinedReplayAgent", "ReplayActionsOnlyAgent", "SimpleReplayAgent"]),
});

export class ReplayEpisode {
  readonly name: string;
  readonly directory: string;
  readonly episode: Episode;
  readonly metrics: {
    [key: string]: number;
  };
  readonly frames: string[];
  readonly agent_details: AgentDetails[];
  readonly action_space: ActionSpace;
  readonly tracks: (Track | TrackGroup)[];
  readonly replay_mismatch: boolean;
  readonly mismatch_details: string[];
  readonly replay_kind: "CombinedReplayAgent" | "ReplayActionsOnlyAgent" | "SimpleReplayAgent";

  public constructor(
    name: string,
    directory: string,
    episode: Episode,
    metrics: {
      [key: string]: number;
    },
    frames: string[],
    agent_details: AgentDetails[],
    action_space: ActionSpace,
    replay_mismatch: boolean,
    mismatch_details: string[],
    replay_kind: "CombinedReplayAgent" | "ReplayActionsOnlyAgent" | "SimpleReplayAgent",
  ) {
    this.name = name;
    this.directory = directory;
    this.episode = episode;
    this.metrics = metrics;
    this.frames = frames;
    this.agent_details = agent_details;
    this.action_space = action_space;
    this.replay_mismatch = replay_mismatch;
    this.mismatch_details = mismatch_details;
    this.replay_kind = replay_kind;
    this.tracks = this.computeTracks();
  }

  public static fromJSON(json: z.infer<typeof ReplayEpisodePayloadSchema>): ReplayEpisode {
    return new ReplayEpisode(
      json.name,
      json.directory ?? json.rundir ?? "",
      json.episode,
      json.metrics,
      json.frames,
      json.agent_details,
      json.action_space,
      json.replay_mismatch,
      json.mismatch_details,
      json.replay_kind,
    );
  }

  private computeTracks() {
    const tracks = [new Track("Rewards", "numeric", this.episode.rewards)] as (
      | Track
      | TrackGroup
    )[];
    const keys = Object.keys(this.agent_details[0]);
    for (const key of keys) {
      // Gather the logs by key across all time steps
      const values = this.agent_details.map((details) => details[key]);
      if (typeof values[0] === "number") {
        tracks.push(new Track(key, "numeric", values as number[]));
      } else if (Array.isArray(values[0]) && typeof values[0][0] === "number") {
        const values2D = values as number[][];
        const group = new TrackGroup(key, []);
        for (let i = 0; i < this.nAgents(); i++) {
          group.subTracks.push(
            new Track(
              `${key} Agent ${i}`,
              "numeric",
              values2D.map((v) => v[i]),
            ),
          );
        }
        tracks.push(group);
      } else {
        const values3D = values as number[][][];
        const group = new TrackGroup(key, []);
        for (let i = 0; i < this.nAgents(); i++) {
          for (let j = 0; j < values3D[0][i].length; j++) {
            group.subTracks.push(
              new Track(
                `${key} Agent ${i}/${j}`,
                "numeric",
                values3D.map((v) => v[i][j]),
              ),
            );
          }
        }
        tracks.push(group);
      }
    }
    return tracks;
  }

  public getTrack(trackLabel: string) {
    for (const track of this.tracks) {
      if (track instanceof TrackGroup) {
        const subTrack = track.getTrack(trackLabel);
        if (subTrack != null) {
          return subTrack;
        }
      } else if (track.label === trackLabel) {
        return track;
      }
    }
  }

  public nAgents() {
    return this.episode.actions[0].length;
  }

  public length() {
    return this.episode.episode_len;
  }

  public frameAt(step: number) {
    return this.frames[step] || "";
  }
}

export interface Episode {
  all_available_actions: boolean[][][];
  all_extras: number[][][];
  all_observations: number[][][] | number[][][][][];
  actions: ActionValue[][];
  episode_len: number;
  is_finished: boolean;
  rewards: number[];
  states: number[][][];
}

export type ActionValue = number | number[];

export interface Transition {
  obs: number[][];
  extras: number[][];
  actions: number[];
  reward: number;
  available_actions: number[][];
  states: number[][];
  qvalues: number[][];
  prev_frame: string;
  current_frame: string;
}
