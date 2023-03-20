<template>
    <div>
        <h3> Layers </h3>
        <div v-for="layer in obs">
            {{ layer }}
        </div>

        <h3> Extras </h3>
        <table class="table table-responsive">
            <tr>
                <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
            </tr>
            <tr>
                <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
            </tr>
        </table>
    </div>
</template>

<script setup lang="ts">

defineProps<{
    obs: number[][][],
    extras: number[]
}>();

</script>