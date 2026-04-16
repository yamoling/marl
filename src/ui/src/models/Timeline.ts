import { z } from "zod";

export const TimelineTrackKindSchema = z.enum(['numeric', 'categorical']);
export type TimelineTrackKind = z.infer<typeof TimelineTrackKindSchema>;

export const TrackConfigSchema = z.object({
    label: z.string(),
    kind: TimelineTrackKindSchema,
});
export type TrackConfig = z.infer<typeof TrackConfigSchema>;


export class Track implements TrackConfig {
    readonly label: string;
    kind: TimelineTrackKind;
    readonly values: number[];

    public constructor(label: string, kind: TimelineTrackKind, values: number[]) {
        this.label = label;
        this.kind = kind;
        this.values = values;

    }

    public length() {
        return this.values.length
    }

    public nDistinctValues() {
        return new Set(this.values).size;
    }
}



export class TrackGroup {
    readonly label: string;
    readonly subTracks: Track[];

    public constructor(label: string, tracks: Track[]) {
        this.label = label;
        this.subTracks = tracks;
    }

    public static fromRaw(label: string, data: number[][]): TrackGroup {
        if (data.length === 0) {
            return new TrackGroup(label, []);
        }
        const tracks = (data as number[][]).map((values, index) => new Track(`${label}/${index + 1}`, 'numeric', values));
        return new TrackGroup(label, tracks);
    }

    public getTracks(): Track[] {
        return this.subTracks.map((subTrack) => (subTrack instanceof TrackGroup) ? subTrack.getTracks() : [subTrack]).flat();
    }

    public getTrack(trackLabel: string): Track | undefined {
        for (const subTrack of this.subTracks) {
            if (subTrack instanceof TrackGroup) {
                const found = subTrack.getTrack(trackLabel);
                if (found) return found;
            }
            else if (subTrack.label === trackLabel) {
                return subTrack;
            }
        }
    }

    public getMajorityKind(): TimelineTrackKind {
        const kinds = this.getTracks().map(t => t.kind);
        const numeric = kinds.filter((kind) => kind === 'numeric').length;
        const categorical = kinds.length - numeric;
        return numeric > categorical ? 'numeric' : 'categorical';
    }

    public setKind(kind: TimelineTrackKind) {
        for (const subTrack of this.subTracks) {
            if (subTrack instanceof TrackGroup) {
                subTrack.setKind(kind);
            }
            else {
                subTrack.kind = kind;
            }
        }
    }
}