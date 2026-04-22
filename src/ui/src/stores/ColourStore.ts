import { defineStore } from "pinia";
import { stringToRGB } from "../utils";
import { computed } from "vue";
import { useSettingsStore } from "./SettingsStore";

export const useColourStore = defineStore("ColourStore", () => {
    const settingsStore = useSettingsStore();
    const colours = computed(() => settingsStore.settings.visualization.colours);

    function get(logdir: string): string {
        let colour = colours.value[logdir];
        if (colour != null) {
            return colour;
        }
        colour = stringToRGB(logdir);
        set(logdir, colour);
        return colour;
    }


    function set(logdir: string, colour: string) {
        settingsStore.setColour(logdir, colour);
    }


    return {
        colours,
        get,
        set
    };
});
