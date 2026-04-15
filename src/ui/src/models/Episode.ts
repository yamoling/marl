import { ActionSpace } from "./Env"
import { Track, TrackGroup } from "./Timeline";


export interface ReplayEpisodeSummary {
    name: string,
    directory: string,
    metrics: {
        [key: string]: number
    }
}

/**
 * Step-wise agent details can be:
 *  - 0D i.e. common to all agents;
 *  - 1D agent-wise scalar (agent-wise state-value estimation, agent-wise option selected, ...);
 *  - 2D agent-wise and <extra dimension>-wise (e.g. q-values, action probabilities, ...).
 */
export type AgentDetails = Record<string, number[] | number[][] | number[][][]>;


export class ReplayEpisode {
    readonly name: string
    readonly directory: string
    readonly episode: Episode
    readonly metrics: {
        [key: string]: number
    }
    readonly frames: string[]
    readonly agent_details: AgentDetails
    readonly action_space: ActionSpace
    readonly tracks: (Track | TrackGroup)[]

    public constructor(
        name: string,
        directory: string,
        episode: Episode,
        metrics: {
            [key: string]: number
        },
        frames: string[],
        agent_details: AgentDetails,
        action_space: ActionSpace
    ) {
        this.name = name;
        this.directory = directory;
        this.episode = episode;
        this.metrics = metrics;
        this.frames = frames;
        this.agent_details = agent_details;
        this.action_space = action_space;
        this.tracks = this.getTracks();
    }

    public static fromJSON(json: any): ReplayEpisode {
        return new ReplayEpisode(
            json.name,
            json.directory,
            json.episode,
            json.metrics,
            json.frames,
            json.agent_details,
            json.action_space
        );
    }

    public getTracks(): (Track | TrackGroup)[] {
        const tracks = [
            new Track("Rewards", "numeric", this.episode.rewards),
            new TrackGroup("Available actions", this.episode.all_available_actions.map((agentActions, agentIndex) => {
                return new Track(`Agent ${agentIndex + 1}`, 'categorical', agentActions.map((actions) => actions.map((a) => a ? 1 : 0)).flat());
            }))
        ];
        return tracks
    }

    public getTrack(trackLabel: string) {
        for (const track of this.tracks) {
            if (track instanceof TrackGroup) {
                const subTrack = track.getTrack(trackLabel);
                if (subTrack != null) {
                    return subTrack;
                }
            }
            else if (track.label === trackLabel) {
                return track;
            }
        }
    }

    public nAgents() {
        return this.episode.actions[0].length
    }

    public length() {
        return this.episode.episode_len;
    }

    public frameAt(step: number) {
        return this.frames[step] || '';
    }
}



export interface Episode {
    all_available_actions: boolean[][][]
    all_extras: number[][][]
    all_observations: number[][][] | number[][][][][]
    actions: ActionValue[][]
    episode_len: number
    is_finished: boolean
    rewards: number[]
    states: number[][][]
}

export type ActionValue = number | number[];

export interface Transition {
    obs: number[][],
    extras: number[][],
    actions: number[],
    reward: number,
    available_actions: number[][],
    states: number[][],
    qvalues: number[][],
    prev_frame: string,
    current_frame: string
}