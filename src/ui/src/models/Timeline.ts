export type TimelineTrackKind = 'continuous-bar' | 'continuous-line' | 'discrete';

export type TimelineBin = {
    key: string;
    startStep: number;
    endStep: number;
    representativeStep: number;
};

export class TimelineModel {
    readonly episodeLength: number;
    readonly desiredBinCount: number;

    constructor(episodeLength: number, desiredBinCount: number) {
        this.episodeLength = Math.max(0, episodeLength);
        this.desiredBinCount = Math.max(1, desiredBinCount);
    }

    public buildBins(transitionCount: number): TimelineBin[] {
        const steps = Math.max(0, transitionCount);
        if (steps === 0) return [];

        const binCount = Math.min(steps, this.desiredBinCount);
        const binSize = Math.ceil(steps / binCount);
        const bins: TimelineBin[] = [];

        for (let index = 0; index < binCount; index++) {
            const startStep = index * binSize + 1;
            const endStep = Math.min(steps, startStep + binSize - 1);
            if (startStep > endStep) break;

            const midpoint = Math.floor((startStep + endStep) / 2);
            bins.push({
                key: `${startStep}-${endStep}`,
                startStep,
                endStep,
                representativeStep: midpoint,
            });
        }

        return bins;
    }

    public nowPercent(currentStep: number): number {
        if (this.episodeLength <= 0) return 0;
        const clampedStep = Math.max(0, Math.min(this.episodeLength, currentStep));
        return (clampedStep / this.episodeLength) * 100;
    }
}

export abstract class TimelineTrack {
    readonly id: string;
    readonly label: string;
    readonly kind: TimelineTrackKind;
    visible: boolean;

    public constructor(id: string, label: string, kind: TimelineTrackKind, visible = true) {
        this.id = id;
        this.label = label;
        this.kind = kind;
        this.visible = visible;
    }
}

export type ContinuousBarCell = TimelineBin & {
    value: number;
    normalized: number;
};

export class ContinuousBarTrack extends TimelineTrack {
    readonly values: number[];

    constructor(id: string, label: string, values: number[], visible = true) {
        super(id, label, 'continuous-bar', visible);
        this.values = values;
    }

    buildCells(bins: TimelineBin[]): ContinuousBarCell[] {
        const absMax = this.absoluteMax();
        return bins.map((bin) => {
            const segment = this.values.slice(bin.startStep - 1, bin.endStep);
            const mean = segment.length > 0
                ? segment.reduce((sum, value) => sum + value, 0) / segment.length
                : 0;

            return {
                ...bin,
                value: mean,
                normalized: mean / absMax,
            };
        });
    }

    private absoluteMax(): number {
        if (this.values.length === 0) return 1;
        const max = Math.max(...this.values.map((value) => Math.abs(value)));
        return max > 0 ? max : 1;
    }
}

export type DiscreteCell = TimelineBin & {
    category: string | null;
    changedFromPrevious: boolean;
};

export class DiscreteTrack extends TimelineTrack {
    readonly values: Array<string | null>;

    constructor(id: string, label: string, values: Array<string | null>, visible = true) {
        super(id, label, 'discrete', visible);
        this.values = values;
    }

    buildCells(bins: TimelineBin[]): DiscreteCell[] {
        const cells = bins.map((bin) => {
            const values = this.values.slice(bin.startStep - 1, bin.endStep);
            const cell: DiscreteCell = {
                ...bin,
                category: dominantCategory(values),
                changedFromPrevious: false,
            };
            return cell;
        });

        for (let index = 1; index < cells.length; index++) {
            cells[index].changedFromPrevious = cells[index - 1].category !== cells[index].category;
        }
        return cells;
    }
}

function dominantCategory(values: Array<string | null>): string | null {
    const counts = new Map<string, number>();
    for (const value of values) {
        if (value == null) continue;
        counts.set(value, (counts.get(value) ?? 0) + 1);
    }

    let selected: string | null = null;
    let maxCount = -1;
    for (const [value, count] of counts.entries()) {
        if (count > maxCount) {
            maxCount = count;
            selected = value;
        }
    }

    return selected;
}
