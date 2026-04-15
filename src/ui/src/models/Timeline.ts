export type TimelineTrackKind = 'numeric' | 'categorical';

export interface TrackConfig {
    label: string;
    kind: TimelineTrackKind;
}

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
    readonly subTracks: Track[] | TrackGroup[];

    public constructor(label: string, tracks: Track[] | TrackGroup[]) {
        this.label = label;
        this.subTracks = tracks;
    }

    public static fromRaw(label: string, data: number[][] | number[][][]): TrackGroup {
        if (data.length === 0) {
            return new TrackGroup(label, []);
        }
        // 2D: the first dimension is the track index, the second dimension is the time dimension
        if (data[0][0] instanceof Number) {
            const tracks = (data as number[][]).map((values, index) => new Track(`${label}/${index + 1}`, 'numeric', values));
            return new TrackGroup(label, tracks);
        }
        const subGroups = (data as number[][][]).map((groupData, index) => TrackGroup.fromRaw(`${label}/${index + 1}`, groupData));
        return new TrackGroup(label, subGroups);
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
        const kindCounts: { [key in TimelineTrackKind]: number } = {
            'numeric': 0,
            'categorical': 0
        };
        for (const subTrack of this.subTracks) {
            if (subTrack instanceof TrackGroup) {
                const subKind = subTrack.getMajorityKind();
                kindCounts[subKind]++;
            }
            else {
                kindCounts[subTrack.kind]++;
            }
        }
        if (kindCounts['numeric'] > kindCounts['categorical']) {
            return 'numeric';
        }
        return 'categorical';
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