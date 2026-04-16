import { defineStore } from "pinia";
import { stringToRGB, qvalueLabelToHSL } from "../utils";
import { ref } from "vue";
import { fromJsonString } from "../utils";

export const useColourStore = defineStore("ColourStore", () => {

    const colours = ref(initColoursFromLocalStorage());

    function initColoursFromLocalStorage() {
        try {
            const entries = JSON.parse(localStorage.getItem("logdirColours") ?? "[]");
            return new Map<string, string>(entries);
        } catch (e) {
            return new Map<string, string>();
        }
    }

    function saveColoursToLocalStorage() {
        localStorage.setItem("logdirColours", JSON.stringify(Array.from(colours.value.entries())));
    }

    function get(logdir: string): string {
        // TODO: store label+flag agent/value-hue
        let colour = colours.value.get(logdir);
        if (colour != null) {
            return colour;
        }
        colour = stringToRGB(logdir);
        set(logdir, colour);
        return colour;
    }

    function getQColour(label: string, qv_or_ag: boolean): string {
        let colour = colours.value.get(label);
        if (colour != null) {
            return colour;
        }
        colour = qvalueLabelToHSL(label, qv_or_ag);
        set(label, colour);
        return colour;
    }

    function set(logdir: string, colour: string) {
        const updated = new Map(colours.value);
        updated.set(logdir, colour);
        colours.value = updated;
        saveColoursToLocalStorage();
    }


    return {
        colours,
        get,
        getQColour,
        set
    };
});
