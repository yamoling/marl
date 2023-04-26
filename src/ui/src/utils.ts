

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

