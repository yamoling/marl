import { defineStore } from "pinia";
import { ref } from "vue";

export interface AppError {
    id: number;
    title: string;
    detail: string;
    errorType: string;  // Python exception class name from backend, e.g. "ValueError"
    timestamp: Date;
    autoDismiss: boolean;
}

export const useErrorStore = defineStore("ErrorStore", () => {
    const errors = ref<AppError[]>([]);
    let nextId = 0;

    function push(title: string, detail: string, errorType: string = "", autoDismiss = true): void {
        errors.value.push({
            id: nextId++,
            title,
            detail,
            errorType,
            timestamp: new Date(),
            autoDismiss,
        });
    }

    function dismiss(id: number): void {
        errors.value = errors.value.filter(e => e.id !== id);
    }

    return { errors, push, dismiss };
});
