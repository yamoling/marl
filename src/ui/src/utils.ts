

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
    let hash = 17;
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
