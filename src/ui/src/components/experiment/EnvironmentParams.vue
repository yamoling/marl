<template>
    <JsonInspector title="Environment" :value="props.env" />
</template>

<script setup lang="ts">
import { Env } from "../../models/Env";
import JsonInspector from "./JsonInspector.vue";

const props = defineProps<{
    env: Env
}>()
</script>