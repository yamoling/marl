<script setup lang="ts">
/** Renders the pending confirmation of the confirm store (single instance). */
import { computed } from "vue";
import { useConfirmStore } from "../../stores/confirm";
import ConfirmDialog from "./ConfirmDialog.vue";

const store = useConfirmStore();
const c = computed(() => store.current);
</script>

<template>
  <ConfirmDialog
    :open="!!c"
    :title="c?.title ?? ''"
    :message="c?.message"
    :confirm-label="c?.confirmLabel"
    :danger="c?.danger"
    :confirm-text="c?.confirmText"
    :busy="store.busy"
    :error="store.error"
    @confirm="store.confirm()"
    @cancel="store.cancel()"
  />
</template>
