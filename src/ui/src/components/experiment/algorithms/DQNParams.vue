<template>
    <div class="accordion">
        <div class="accordion-item">
            <h2 class="accordion-header">
                <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDQN"
                    aria-expanded="true" aria-controls="collapseDQN">
                    <b>Algorithm</b>
                </button>
            </h2>
            <div id="collapseDQN" class="accordion-collapse collapse show">
                <div class="accordion-body">
                    <table class="table table-sm">
                        <tbody>
                            <tr>
                                <th>Name</th>
                                <td>{{ algo.name }}</td>
                            </tr>
                            <tr>
                                <th> Qnetwork</th>
                                <td>{{ algo.qnetwork.name }}</td>
                            </tr>
                            <tr>
                                <th class="align-middle"> Train policy </th>
                                <td>
                                    {{ algo.train_policy.epsilon.name.replace("Schedule", "") }} ε-greedy<br />
                                    {{ algo.train_policy.epsilon.start_value }}
                                    <font-awesome-icon :icon="['fas', 'arrow-right']" />
                                    {{ algo.train_policy.epsilon.end_value }}
                                    ({{ algo.train_policy.epsilon.n_steps / 1000 }}k steps)

                                </td>
                            </tr>
                            <tr>
                                <th class="align-middle"> Test policy </th>
                                <td>
                                    {{ algo.test_policy.epsilon.name.replace("Schedule", "") }}
                                    <template v-if="algo.test_policy.epsilon.name.startsWith('Constant')">
                                        ε = {{ algo.test_policy.epsilon.start_value }}
                                    </template>
                                    <template v-else>
                                        ε-greedy<br />
                                        {{ algo.test_policy.epsilon.start_value }}
                                        <font-awesome-icon :icon="['fas', 'arrow-right']" />
                                        {{ algo.train_policy.epsilon.end_value
                                        }}
                                        ({{ algo.test_policy.epsilon.n_steps / 1000 }}k steps)
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
                            <tr>
                                <th>Learning rate</th>
                                <td>{{ trainer.lr.toExponential() }}</td>
                            </tr>

                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>

import { DQN } from "../../../models/Algorithm"
import { Trainer } from "../../../models/Trainer"

defineProps<{
    algo: DQN
    trainer: Trainer
}>();

</script>