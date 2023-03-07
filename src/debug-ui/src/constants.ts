export const HOST = "0.0.0.0";
export const HTTP_PORT = 5174;
export const HTTP_URL = `http://${HOST}:${HTTP_PORT}`;
export function wsURL(port: number) {
    return `ws://${HOST}:${port}`
}
