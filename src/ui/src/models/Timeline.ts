export type TimelineTrackKind = 'numeric' | 'categorical';
export type TimelineTrackValue = number | string | null;

export class TimelineTrack {
    readonly id: string;
    readonly label: string;
    readonly kind: TimelineTrackKind;
    readonly values: TimelineTrackValue[];

    constructor(id: string, label: string, kind: TimelineTrackKind, values: TimelineTrackValue[]) {
        this.id = id;
        this.label = label;
        this.kind = kind;
        this.values = values;
    }

    public length() {
        return this.values.length
    }
}
