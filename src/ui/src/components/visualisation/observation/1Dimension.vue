<template>
    <table class="table table-responsive">
        <tr>
            <th :colspan="obs.length" style="background-color: beige;">Observation</th>
            <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
        </tr>
        <tr>
            <td style="background-color: beige;" v-for="o in obs">
                {{ o.toFixed(3) }}
            </td>
            <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
        </tr>
    </table>
</template>

<script setup lang="ts">

defineProps<{
    obs: number[],
    extras: number[]
}>();

</script>