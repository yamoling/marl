

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
