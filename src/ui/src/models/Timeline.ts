export type TimelineTrackKind = 'continuous-bar' | 'continuous-line' | 'discrete';

export type TimelineBin = {
    key: string;
    startStep: number;
    endStep: number;
    representativeStep: number;
};

export class TimelineModel {
    readonly episodeLength: number;

    constructor(episodeLength: number) {
        this.episodeLength = Math.max(0, episodeLength);
    }

    public buildBins(transitionCount: number, maxBins = transitionCount): TimelineBin[] {
        const steps = Math.max(0, transitionCount);
        const targetBins = Math.max(1, Math.floor(maxBins));
        if (steps === 0) return [];

        if (targetBins >= steps) {
            const bins: TimelineBin[] = [];

            for (let step = 1; step <= steps; step++) {
                bins.push({
                    key: `${step}`,
                    startStep: step,
                    endStep: step,
                    representativeStep: step,
                });
            }

            return bins;
        }

        const bins: TimelineBin[] = [];
        const binSize = Math.ceil(steps / targetBins);

        for (let step = 1; step <= steps; step += binSize) {
            const endStep = Math.min(steps, step + binSize - 1);
            bins.push({
                key: `${step}-${endStep}`,
                startStep: step,
                endStep,
                representativeStep: Math.round((step + endStep) / 2),
            });
        }

        return bins;
    }

    public nowPercent(currentStep: number): number {
        if (this.episodeLength <= 0) return 0;
        const clampedStep = Math.max(0, Math.min(this.episodeLength, currentStep));
        if (clampedStep === 0) return 0;

        // Align to the center of the selected step cell instead of its boundary.
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
    readonly aggregation: 'sum' | 'mean';

    constructor(id: string, label: string, values: number[], visible = true, aggregation: 'sum' | 'mean' = 'sum') {
        super(id, label, 'continuous-bar', visible);
        this.values = values;
        this.aggregation = aggregation;
    }

    buildCells(bins: TimelineBin[]): ContinuousBarCell[] {
        const absMax = this.absoluteMax();
        return bins.map((bin) => {
            const window = this.values.slice(bin.startStep - 1, bin.endStep);
            const sum = window.reduce((accumulator, entry) => accumulator + (entry ?? 0), 0);
            const value = this.aggregation === 'mean' && window.length > 0
                ? sum / window.length
                : sum;

            return {
                ...bin,
                value,
                normalized: value / absMax,
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
    colour: string | null;
};

export class DiscreteTrack extends TimelineTrack {
    readonly values: Array<string | null>;
    private readonly colourByCategory = new Map<string, string>();
    private nextPaletteIndex = 0;

    private static readonly DISTINCT_COLOURS = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        '#393b79',
        '#637939',
        '#8c6d31',
        '#843c39',
        '#7b4173',
        '#3182bd',
    ];

    constructor(id: string, label: string, values: Array<string | null>, visible = true) {
        super(id, label, 'discrete', visible);
        this.values = values;
    }

    buildCells(bins: TimelineBin[]): DiscreteCell[] {
        const cells = [] as DiscreteCell[];

        for (const bin of bins) {
            const window = this.values.slice(bin.startStep - 1, bin.endStep);
            const category = mostFrequentCategory(window);
            const cell: DiscreteCell = {
                ...bin,
                category,
                colour: category == null ? null : this.colourForCategory(category),
            };
            cells.push(cell);
        }

        return cells;
    }

    private colourForCategory(category: string): string {
        const existing = this.colourByCategory.get(category);
        if (existing != null) return existing;

        let colour: string;
        if (this.nextPaletteIndex < DiscreteTrack.DISTINCT_COLOURS.length) {
            colour = DiscreteTrack.DISTINCT_COLOURS[this.nextPaletteIndex];
            this.nextPaletteIndex += 1;
        } else {
            colour = randomHexColour();
        }

        this.colourByCategory.set(category, colour);
        return colour;
    }
}

function randomHexColour(): string {
    const channel = () => Math.floor(Math.random() * 206) + 25;
    const toHex = (value: number) => value.toString(16).padStart(2, '0');
    const red = channel();
    const green = channel();
    const blue = channel();
    return `#${toHex(red)}${toHex(green)}${toHex(blue)}`;
}

function mostFrequentCategory(values: Array<string | null>): string | null {
    const counts = new Map<string, number>();

    for (const value of values) {
        if (value == null) continue;
        counts.set(value, (counts.get(value) ?? 0) + 1);
    }

    let bestCategory: string | null = null;
    let bestCount = 0;

    for (const [category, count] of counts) {
        if (count > bestCount) {
            bestCategory = category;
            bestCount = count;
        }
    }

    return bestCategory;
}
