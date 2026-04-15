export function formatNumber(value: number | string): string {
    const numericValue = typeof value === 'string' ? Number.parseFloat(value) : value;

    if (Number.isNaN(numericValue)) {
        return String(value);
    }

    if (numericValue === Math.floor(numericValue)) {
        return numericValue.toString();
    }

    return numericValue.toFixed(3);
}