import { ExperimentResults } from "./models/Experiment";

/**
 * Compute the shape of a multi-dimensional array.
 */
export function computeShape(array: any[]): number[] {
    const result = [];
    let a = array;
    while (Array.isArray(a)) {
        result.push(a.length);
        a = a[0];
    }
    return result;
}

/**
 * Exponential moving average
 * @param data 
 * @param weight bewteen 0 and 1
 */
export function EMA(data: number[], weight: number) {
    let prevEMA = data[0];
    const result = new Array(data.length);
    for (let i = 0; i < data.length; i++) {
        const newValue = (prevEMA * weight + data[i] * (1 - weight));
        prevEMA = newValue;
        result[i] = newValue;
    }
    return result;
}

/**
 * Checks whether the search string matches the target string.
 * There is a match if all the letters from the search string appear in the target string in the same order.
 * @param search 
 * @param matchWith 
 */
export function searchMatch(search: string, matchWith: string): boolean {
    if (search.length === 0) return true;
    search = search.replaceAll(" ", "");
    let searchIndex = 0;
    for (let i = 0; i < matchWith.length; i++) {
        // if (/\s/g.test(search[searchIndex])) {
        //     searchIndex++;
        // }
        if (matchWith[i] === search[searchIndex]) {
            searchIndex++;
            if (searchIndex === search.length) return true;
        }
    }
    return false;
}

function hashString(s: string) {
    let hash = 31;
    for (let i = 0; i < s.length; i++) {
        const chr = s.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};


export function stringToRGB(s: string) {
    const hash = hashString(s);
    let colour = '#';
    for (let i = 0; i < 3; i++) {
        let value = (hash >> (i * 8)) & 0xFF;
        colour += value.toString(16).padStart(2, '0');
    }
    return colour;
}


export function downloadStringAsFile(textToSave: string, fileName: string) {
    const hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
    hiddenElement.target = '_blank';
    hiddenElement.download = fileName;
    hiddenElement.click();
}




export function confidenceInterval(mean: number[], std: number[], nSamples: number, confidence: number) {
    const sqrtN = Math.sqrt(nSamples);
    const lower = Array<number>(std.length);
    const upper = Array<number>(std.length);
    for (let i = 0; i < std.length; i++) {
        upper[i] = mean[i] + confidence * std[i] / sqrtN;
        lower[i] = mean[i] - confidence * std[i] / sqrtN;
    }
    return { lower, upper };
}

export function clip(values: number[], min: number[], max: number[]) {
    const result = new Array(values.length);
    for (let i = 0; i < values.length; i++) {
        result[i] = Math.min(Math.max(values[i], min[i]), max[i]);
    }
    return result;
}



export function unionXTicks(allXTicks: number[][]) {
    const resTicks = new Set<number>();
    for (const ticks of allXTicks) {
        for (const tick of ticks) {
            resTicks.add(tick);
        }
    }
    return Array.from(resTicks).sort((a, b) => a - b);
}


function interpolate(data: number[], ticks: number[], desiredTicks: number[]) {
    const maxTick = ticks[ticks.length - 1]
    let originIndex = 0;
    let index = 0;
    const res = [];
    while (index < desiredTicks.length && desiredTicks[index] <= maxTick) {
        const desiredTick = desiredTicks[index];
        // Find the closest origin tick that is smaller than the desired tick
        while (ticks[originIndex + 1] <= desiredTick) {
            originIndex++;
        }

        // Easy case: when the tick is the same as the current tick, then push the exact value
        if (ticks[originIndex] === desiredTick) {
            res.push(data[originIndex]);
        }
        // More complicated case: when the required tick does not exist
        else {
            if (originIndex == 0) {
                res.push(data[0]);
            }
            // More complicated case: when the required tick does not exist, then interpolate
            else {
                const prevTick = ticks[originIndex];
                const nextTick = ticks[originIndex + 1];
                const ratio = (desiredTick - prevTick) / (nextTick - prevTick);
                res.push(data[originIndex] + ratio * (data[originIndex] - data[originIndex + 1]));
            }
        }
        index++;
    }
    return res;
}

/*
export function alignTicks(experimentResults: ExperimentResults, desiredTicks: number[]): ExperimentResults {
    // Filter train datasets based on the filtered ticks
    const filteredTrain = experimentResults.train.map(dataset => ({
        ...dataset,
        mean: interpolate(dataset.mean, experimentResults.train_ticks, desiredTicks),
        std: interpolate(dataset.std, experimentResults.train_ticks, desiredTicks),
        min: interpolate(dataset.min, experimentResults.train_ticks, desiredTicks),
        max: interpolate(dataset.max, experimentResults.train_ticks, desiredTicks),
        ci95: interpolate(dataset.ci95, experimentResults.train_ticks, desiredTicks),
    }));

    // Filter test datasets based on the filtered ticks
    const filteredTest = experimentResults.test.map(dataset => ({
        ...dataset,
        mean: interpolate(dataset.mean, experimentResults.test_ticks, desiredTicks),
        std: interpolate(dataset.std, experimentResults.test_ticks, desiredTicks),
        min: interpolate(dataset.min, experimentResults.test_ticks, desiredTicks),
        max: interpolate(dataset.max, experimentResults.test_ticks, desiredTicks),
        ci95: interpolate(dataset.ci95, experimentResults.test_ticks, desiredTicks),
    }));

    // Return a new ExperimentResults with filtered data
    return {
        ...experimentResults,
        test_ticks: [...desiredTicks],
        train: filteredTrain,
        test: filteredTest,
    };
}
*/