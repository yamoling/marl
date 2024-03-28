<template>
    <div class="accordion">
        <div class="accordion-item">
            <h2 class="accordion-header">
                <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseEnv"
                    aria-expanded="true" aria-controls="collapseEnv">
                    <h5>Environment: {{ env.name }}</h5>
                </button>
            </h2>
            <div id="collapseEnv" class="accordion-collapse collapse show">
                <div class="accordion-body">
                    <table class="table table-sm">
                        <tbody>
                            <template v-for="[name, params] in wrappers">
                                <tr>
                                    <th class="align-middle">{{ name }}</th>
                                    <template v-if="params.size > 0">
                                        <td>
                                            <template v-for="[key, value] in params">
                                                {{ key }} = {{ value }}<br>
                                            </template>
                                        </td>
                                    </template>

                                    <td v-else> / </td>
                                </tr>
                            </template>
                            <tr>
                                <th>Observation shape</th>
                                <td>{{ env.observation_shape }}</td>
                            </tr>
                            <tr v-if="env.reward_space.size > 1">
                                <th> Rewards </th>
                                <td>
                                    <!-- Show all labels from env.reward_space.labels -->
                                    <template v-for="(label, i) in env.reward_space.labels">
                                        {{ label }}
                                        <template v-if="i < env.reward_space.size - 1">, </template>
                                    </template>
                                </td>
                            </tr>
                            <tr>
                                <th>Extras shape</th>
                                <td>{{ env.extra_feature_shape }}</td>
                            </tr>
                            <tr>
                                <th> State shape</th>
                                <td> {{ env.state_shape }}</td>
                            </tr>
                            <tr>
                                <th> # Agents</th>
                                <td> {{ env.n_agents }}</td>
                            </tr>
                            <tr>
                                <th class="align-middle"> Actions</th>
                                <td>
                                    <template v-for="(action, i) in env.action_space.action_names">
                                        {{ i }} <font-awesome-icon :icon="['fas', 'arrow-right']" /> {{ action }}
                                        <br>
                                    </template>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { Env, EnvWrapper, getWrapperParameters } from "../../models/Env";

const props = defineProps<{
    env: Env
}>()

const wrappers = computed(() => {
    const wrappers = new Map<string, Map<string, any>>();
    let env = props.env;
    while (Object.hasOwn(env, "full_name")) {
        const wrapper = env as EnvWrapper;
        const name = wrapper.full_name.split('(')[0];
        const params = getWrapperParameters(wrapper);
        wrappers.set(name, params);
        env = wrapper.wrapped;
    }
    return wrappers;
});
</script>