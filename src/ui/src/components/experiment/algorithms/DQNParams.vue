<template>
    <div class="accordion">
        <div class="accordion-item">
            <h2 class="accordion-header">
                <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDQN"
                    aria-expanded="true" aria-controls="collapseDQN">
                    <h5>Algorithm: {{ algoName }}</h5>
                </button>
            </h2>
            <div id="collapseDQN" class="accordion-collapse collapse show">
                <div class="accordion-body">
                    <table class="table table-sm">
                        <tbody>
                            <tr>
                                <th class="align-middle"> Qnetwork</th>
                                <td>
                                    {{ algo.qnetwork.name }} <br>
                                    <template v-if="trainer.grad_norm_clipping">
                                        Grad norm clip={{ trainer.grad_norm_clipping }} <br>
                                    </template>
                                    {{ trainer.target_updater.name }}
                                </td>
                            </tr>
                            <tr v-if="trainer.ir_module">
                                <th>Intrinsic reward</th>
                                <td>{{ trainer.ir_module.name }}</td>
                            </tr>
                            <tr>
                                <th class="align-middle"> Train policy </th>
                                <td>
                                    {{ algo.train_policy.epsilon.name.replace("Schedule", "") }} ε-greedy<br />
                                    {{ algo.train_policy.epsilon.start_value }}
                                    <font-awesome-icon :icon="['fas', 'arrow-right']" />
                                    {{ algo.train_policy.epsilon.end_value }}
                                    <template v-if="algo.train_policy.epsilon.n_steps">
                                        ({{ algo.train_policy.epsilon.n_steps / 1000 }}k steps)
                                    </template>

                                </td>
                            </tr>
                            <tr>
                                <th class="align-middle"> Test policy </th>
                                <td>
                                    {{ algo.test_policy.name }}
                                    <template v-if="'epsilon' in algo.test_policy">
                                        <!-- Epsilon greedy -->
                                        <template v-if="algo.test_policy.epsilon.name.startsWith('Constant')">
                                            ε = {{ algo.test_policy.epsilon.start_value }}
                                        </template>
                                        <template v-else>
                                            ε-greedy<br />
                                            {{ algo.test_policy.epsilon.start_value }}
                                            <font-awesome-icon :icon="['fas', 'arrow-right']" />
                                            {{ algo.train_policy.epsilon.end_value
                                            }}
                                            <template v-if="algo.test_policy.epsilon.n_steps">
                                                ({{ algo.test_policy.epsilon.n_steps / 1000 }}k steps)
                                            </template>

                                        </template>
                                    </template>




                                </td>
                            </tr>
                            <tr>
                                <th>Discount factor</th>
                                <td>{{ trainer.gamma }}</td>
                            </tr>
                            <tr>
                                <th> Update every</th>
                                <td>
                                    {{ trainer.update_interval }}
                                    {{ (trainer.update_on_steps) ? "steps" : "episodes" }}
                                </td>
                            </tr>
                            <tr>
                                <th>Batch size</th>
                                <td>{{ trainer.batch_size }}</td>
                            </tr>
                            <tr v-if="trainer.lr">
                                <th>Learning rate</th>
                                <td>{{ trainer.lr.toExponential() }}</td>
                            </tr>
                            <tr>
                                <th> Memory </th>
                                <td> Size={{ trainer.memory.max_size / 1000 }}k</td>
                            </tr>

                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>

import { computed } from "vue";
import { DQN } from "../../../models/Algorithm"
import { Trainer } from "../../../models/Trainer"

const props = defineProps<{
    algo: DQN
    trainer: Trainer
}>();
const algoName = computed(() => {
    if (props.trainer.mixer) {
        return props.trainer.mixer.name
    } else {
        return props.algo.name
    }
})

</script>