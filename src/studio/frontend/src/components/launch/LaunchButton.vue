<script setup lang="ts">
/**
 * "▶ Start runs", enabled only when `capabilities.launch` is true. While unknown it triggers the
 * lazy health check and shows "checking…"; when false the tooltip quotes the blocking issue.
 */
import { computed, onMounted, watch } from "vue";
import { vTip } from "../../composables/tooltip";
import { launchDisabledReason } from "../../domain/capabilities";
import { useExperimentsStore } from "../../stores/experiments";
import { useUiStore } from "../../stores/ui";
import Icon from "../shell/Icon.vue";

const props = withDefaults(defineProps<{ id: string; small?: boolean }>(), { small: false });
const experiments = useExperimentsStore();
const ui = useUiStore();

const detail = computed(() => experiments.detail(props.id));
const checking = computed(() => !!experiments.checking[props.id]);
const reason = computed(() => launchDisabledReason(detail.value?.capabilities, detail.value?.issues ?? [], checking.value));
const enabled = computed(() => detail.value?.capabilities.launch === true);

onMounted(() => experiments.ensureHealth(props.id));
watch(() => [props.id, !!detail.value] as const, () => experiments.ensureHealth(props.id));
</script>

<template>
  <span v-tip="reason" class="launch-wrap">
    <button type="button" class="btn primary" :class="{ small }" :disabled="!enabled" data-act="start-runs" @click="ui.launchFor = id">
      <Icon name="play" :size="12" />{{ detail?.capabilities.launch === null ? "Start runs (checking…)" : "Start runs" }}
    </button>
  </span>
</template>

<style scoped>
.launch-wrap {
  display: inline-flex;
}
</style>
