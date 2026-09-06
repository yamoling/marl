# Algorithm and optimization audit

Date: 2026-09-05

The audit found defects in the Bellman targets, temporal returns, replay buffers, recurrent state handling, policy losses, and training lifecycle. The fixes and 46 new regression cases are in the working tree. The final suite passes **532 tests, with one skip**, compared with **486 passing tests and one skip** before the changes. No commits were created.

Several legacy implementations still need substantial work. Passing the suite does **not** establish that DDPG, HAVEN, AlphaZero, QTRAN, or every hierarchical/continuous/multi-objective configuration is usable. Those gaps are listed below rather than represented as repaired algorithms.

## Scope and method

I inspected the trainer implementations in `src/marl/algos/`, their intrinsic-reward and MAVEN submodules, the mixing networks, policy and agent implementations, recurrent/model-bank code, batch and replay abstractions, and the runner's initialization/evaluation lifecycle. I also inspected logging, result aggregation, environment shaping, and representative UI/backend code for optimization opportunities. This was an implementation audit, not a line-by-line security review of the frontend, a formal verification, or an empirical reproduction of every paper.

The checks concentrate on algorithm mechanics rather than hyperparameter choices: which action is selected, which value is evaluated, where gradients flow, whether episode boundaries stop traces, whether padding contributes to a loss, and whether replay indices retain their meaning.

The new cases are in [test_algorithm_audit.py](../tests/test_algorithm_audit.py). I ran groups of regressions against the implementation before correcting them. These exposed failures such as NaN terminal targets, wrong return values, overwritten utilities, invalid tensor dimensions, missing optimizer groups, and construction errors. Additional integration checks cover the repaired paths. Existing tests for n-step availability and recurrent acting state were adjusted where they encoded the old incorrect behavior.

The existing untracked `ACER multi-agent implementation.md`, `doc/lan.md`, `reports/profiling/`, and `todo.md` were preserved. Existing experiment results were not rewritten. The repository's actual `README.md` was read; there is no lowercase `readme.md` in this checkout.

## Corrections

### Bellman targets and value factorization

| Area | Defect and consequence | Correction and evidence |
|---|---|---|
| Tabular Q-learning | Calling the generated parent constructor from `__post_init__` recursed. The update also indexed actions incorrectly, bootstrapped terminal transitions, and maximized over illegal actions. | Initialize the parent once; update each agent's chosen table entry; bootstrap only legal successors of nonterminal transitions. Numerical regression checks chosen and unchosen entries. |
| Tabular persistence | Save wrote `qlearning.pkl`; load looked for `qtable.pkl`. | Load the saved filename, with a fallback for the legacy name. Round-trip regression included. |
| Greedy, epsilon-greedy, categorical policies | Masking wrote `-inf` into the caller's Q-values. A tabular agent passes its actual table entry, so action selection could permanently corrupt learning state. | Mask into a separate array. Three regressions require the input utilities to remain identical. |
| Ordinary DQN | In non-double mode, action masking modified the target utilities themselves. An all-unavailable terminal successor produced `-inf * 0 = NaN`. | Separate action-selection values from evaluation values; explicitly zero terminal/padded bootstrap values. Both double and ordinary modes are tested. |
| Multi-objective DQN | The trainer gathered/maximized along the objective axis as if it were the action axis. Individual reward expansion also interleaved objectives incorrectly. | Select an action using summed objective utilities, consistent with the acting agent, then gather all objectives of that action. Preserve objective order and objective-shaped transition masks. A two-objective update is tested. This does not certify every multi-objective mixer/recurrent combination. |
| QPLEX | The mixer's positional signature conflicted with the common mixer interface. Target mixing raised `NotImplementedError` because next actions were missing. | Accept state extras through the common interface. Supply the exact target action selected by DQN, including the online network's choice in Double DQN. Tests perform full gradient updates in both modes and explicitly distinguish online and target argmax actions. |
| Single-agent one-hot actions | Squeezing the final action axis removed the agent dimension when there was one agent. | Preserve the agent axis when constructing one-hot actions. |
| QATTEN | Flattening from dimension 1 merged episode and state features, so episode batches failed. | Flatten only leading batch dimensions; compare episode-shaped output with the equivalent flat batch. Also replace the diagonal value matrix with scalar attention values. |
| Noisy QMLP | The head used `hidden_sizes[1]` and `n_actions`, which fails with one hidden layer and omits the dueling value output. | Use the last hidden width and the declared output size. Test noisy networks with and without dueling. |
| Q-network actor adapter | The adapter supplied `logits()`, but categorical policy construction calls `forward()`. | Implement `forward()` and retain `logits()` as a compatibility alias. Test actual distribution construction. |
| MLP output activation | A requested output activation was replaced by the hidden activation. | Construct the requested final activation; a sigmoid-versus-ReLU regression catches the difference. |

### Returns and on-policy training

| Area | Defect and consequence | Correction and evidence |
|---|---|---|
| Monte Carlo bootstrap | A multidimensional final value was sliced with `[-1]`, silently broadcasting one episode's bootstrap into other episodes. | Preserve the complete episode/agent shape. Regression uses distinct values for each episode and agent. |
| Time-limit boundaries | GAE stopped only at true termination. A truncated transition followed by an environment reset could propagate rewards from the next episode. | Introduce trace boundaries that include truncation, while retaining nonterminal TD bootstrapping. Numerical example previously returned 78 instead of 3. |
| Unequal episode lengths | A single value from the final padded batch row could not bootstrap a shorter truncated episode correctly. | Allow per-transition successor values in the Monte Carlo recursion; PPO supplies them. Padding is excluded. Regression checks a short truncated episode beside a longer terminated episode. |
| PPO critic inputs | Successor observations were paired with current extras. | Pair `next_obs` with `next_extras`. Recurrent critics unroll the full observation history before slicing successor values. |
| PPO/PPOC epochs | Each nominal epoch sampled only one minibatch. Some rollout items were never used, and the number of optimizer steps did not match the epoch definition. | Shuffle the full rollout per epoch and visit every minibatch, including a short final minibatch. PPO regression verifies exact coverage for five transitions, minibatches of two, and two epochs. |
| PPO padding/statistics | Padding diluted approximate KL, and the MSE branch did not explicitly mask its error. Variable-sized final minibatches also require compatible log aggregation. | Mask KL and MSE contributions; flatten log samples before aggregating them. |
| PPOC initialization | The parent-constructor call failed with its `InitVar` arguments. The transition-memory selector checked `"transition"`, although the trainer uses `"step"`. | Correct initialization and select transition memory for step intervals. |
| PPOC value target | Normalized actor advantages were added to values to form critic targets, changing the value-learning objective when normalization was enabled. | Form value targets from raw GAE, then normalize only actor advantages. Regression uses unequal rewards. |
| PPOC mixer gradient | The critic loss used the target mixer, which was outside the optimizer, instead of training the online mixer. | Use the online mixer. Regression requires online mixer gradients and no target mixer gradients. |
| PPOC termination/entropy | Termination masks repeated the agent axis even when it was already present, and reshaped away episode dimensions. Entropy included padding. | Broadcast masks only when needed; average valid termination terms and mask entropy/policy terms. An episode-shaped termination regression checks the numerical result. |
| Option-Critic | Construction failed, and its optimizer omitted mixer parameters. | Correct parent initialization and include the online mixer in the optimizer. Constructor/parameter-membership regression included. |

The PPO epoch correction follows the full-data, multiple-epoch update described in [PPO, Algorithm 1](https://arxiv.org/pdf/1707.06347). It changes the amount of optimization at the same configuration values; it is not a learning-rate adjustment.

### Replay buffers

**N-step replay.** The previous buffer changed rewards without consistently changing the successor observation/state. It withheld too many transitions, failed to expose complete terminal tails, and provided no matching n-step discount to DQN. Its inherited `add_transition()` bypassed its own accumulation code.

The buffer now holds an unfinished trajectory separately. After `n` rewards it emits a complete transition with the accumulated reward, the correct successor observation and state, terminal/truncation flags, and `bootstrap_discount = gamma**actual_horizon`. Episode ends flush shortened tails. DQN consumes the stored discount. Tests check both the stored transition and the actual Bellman target; with three unit rewards, gamma 0.5, and successor value 8, the target is 2.75.

**Prioritized replay.** The sum tree uses physical ring slots; the wrapped deque exposes logical oldest-first indices. After eviction those indices referred to different experiences. In addition, the transition/episode insertion hooks did nothing, clearing did not clear the wrapped memory/tree, and per-item importance weights did not fit individual-agent loss tensors.

The implementation now tracks the next physical slot and translates sampled indices after eviction. Insertion and clear operate on both structures. DQN broadcasts importance weights over agent/time dimensions. Priority updates reduce errors to one scalar per sampled item, using the maximum absolute error across remaining dimensions after the existing optional objective averaging. Schedules receive the supplied time step. Tests cover eviction, insertion through the trainer-facing hook, a prioritized individual-DQN update, and clearing.

The general sampling and correction mechanism is described in [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952). The maximum aggregation across agents/episode time is an explicit multi-agent implementation choice, not a claim that the single-agent paper specifies it.

### Recurrent state and lifecycle

- Episode-batched RNN forwards now start from reset and preserve the hidden state used by the acting agent. Previously, a different batch size could raise a hidden-state shape error, or successive training calls could share an autograd history. The regression checks repeatability, preservation of acting state, and separate backward passes.
- Recurrent Q-network batch forwarding saves/restores nested recurrent states, including on exceptions. Repeated evaluation-mode calls no longer overwrite the saved training state. This matters because the runner and agents can both request evaluation mode.
- ACER preserves the time/episode structure for recurrent actors and critics. Successor evaluation includes the initial observation prefix. A recurrent ACER regression performs two consecutive updates.
- The recurrent categorical actor was missing its dataclass decorator, so its factory's declared layer arguments were rejected. The critic factory also swapped vector and image recurrent classes. These construction paths were corrected.
- DQN, PPO, and ACER preserve optimizer moments across `.to()` calls. DQN also preserves auxiliary parameter groups, which MAVEN adds for its discriminator and trajectory aggregator. Tests check optimizer history and an extra group.
- Trainer randomization synchronizes directly registered target-updater parameter pairs. The simple runner no longer randomizes the shared acting network again after trainer randomization. This removes an initialization mismatch for directly owned target networks. Nested hierarchical initialization remains a separate gap below.
- DQN checkpoints store target networks under `dqn-targets/` and leave online weights in the existing root location. Older checkpoints without that directory initialize targets from the loaded online weights. Mixer loading now uses the same `mixer.weights` filename as saving. Round-trip testing keeps distinct online and target values.
- The runner truncates on the transition that reaches the requested step budget, rather than taking an extra transition. A three-step budget is checked against a twenty-step environment.

### Intrinsic rewards

**ICM.** Construction assigned a child module before initializing `torch.nn.Module`; `.to()` referenced nonexistent `_feature`; inverse cross-entropy received already-softmaxed probabilities; and shape handling assumed transition batches. The encoder is now registered after module initialization, device movement uses the registered modules, inverse loss takes logits, and forward/inverse losses preserve leading episode dimensions and exclude padding. The image factory uses a CNN with an output projection of the requested feature size. A numerical cross-entropy regression and an episode-shaped update cover the core loss path.

The inverse-logit correction was checked against the [authors' ICM implementation](https://raw.githubusercontent.com/pathak22/noreward-rl/master/src/model.py). Its forward feature-gradient convention is retained; no unsupported claim about a required stop-gradient was used to change that objective.

**RND.** An empty random update mask divided by zero. Sampling masked feature coordinates rather than complete samples. Episode normalization treated episode indices as feature dimensions, and reward normalization used extrinsic returns; zero extrinsic rewards could make positive curiosity infinite. The image encoder also used an obsolete constructor.

RND now subsamples valid samples, skips empty updates, pools valid state samples across leading batch dimensions, and normalizes using discounted intrinsic-reward statistics with a denominator floor. Padding contributes neither statistics nor output reward. The image encoder has a feature projection. Tests cover zero update ratio, image input, episode statistic shapes, positive curiosity with zero extrinsic reward, and padding. These checks do not reproduce the original RND Atari experiments or validate every multi-objective configuration.

PPO and ACER now explicitly broadcast scalar intrinsic rewards across individual-agent reward axes. Advantage and value-potential intrinsic rewards suppress terminal bootstrap values; the advantage critic's invalid boolean subtraction was also corrected.

## Remaining implementation gaps

These findings remain open. They are primarily legacy implementations, unsupported combinations, or changes requiring a coherent redesign across several interfaces. They should be addressed before treating those paths as validated baselines.

| Priority | Location | Finding and required follow-up |
|---|---|---|
| High | `algos/ddpg.py` | The trainer refers to nonexistent `self.network`, uses obsolete actor/critic calls, and has no functioning target-network implementation. Its categorical-actor/state-value interfaces do not supply the deterministic actor/action-value critic required for DDPG. This needs an implemented continuous-control contract and an end-to-end regression, not a renamed attribute. |
| High | `agents/mcts/alphazero.py`, `alpha_node.py` | Actor and critic interfaces are mixed: expansion asks the actor for value estimates; training unpacks actor output as policy and value; targets store only the chosen child's probability. Backpropagation does not consistently reconstruct discounted transition returns. Rebuild around separate actor/critic calls, full visit-count distributions, and a stated value convention. |
| High | `nn/mixers/qtran.py` | Explicitly unfinished, with additional stale constructor/attribute references. The displayed optimality-constraint expression also does not square the entire residual. It is not an operational QTRAN trainer. |
| High | `algos/haven.py` | `_episode_step` is not initialized; meta-episode construction explicitly raises; reward-window accounting omits boundary rewards. The meta-transition horizon also needs a consistent semi-MDP discount. The hierarchical rollout contract must be repaired together with the learner. |
| High | `algos/reinforce.py` | References an unavailable `ActorCritic` abstraction. TD1 bootstrapping is unmasked, Monte Carlo returns are created on CPU, available-action masks are not passed to the training distribution, and the baseline has no value-fitting loss. A modern actor/critic adapter and training regression are still needed. |
| High | `nn/model_bank/options.py`, `algos/option_critic.py` | Option-network factories still contain obsolete class names/constructor calls. The on-policy option selector has list/tensor conventions that are not consistently compatible with batched CNN inputs. Fixing the trainer's initialization and optimizer does not establish a working end-to-end option baseline. |
| High | `models/trainer.py`, `algos/maven/`, `agents/hierarchical/` | Generic hierarchical randomization flattens child networks but does not invoke each child's target/average-policy synchronization. The MAVEN worker does not receive the declared `lr`, and `ExpectedReturnTrainer.undiscounted` is not used. Repeated testing calls can overwrite saved episode noise. The repaired optimizer-group preservation and Q-to-actor adapter address only part of this integration. |
| Medium | `algos/ppoc.py` | The documentation calls its objective “dual-clipped,” but the implementation supplies ordinary PPO clipping. The epoch-count metric still reflects the loop/update index rather than a clean count of completed epochs. Padding treatment for approximate KL needs the same care as PPO. |
| Medium | Multi-objective episode paths | `EpisodeBatch.multi_objective()` remains unimplemented. Objective/agent broadcasting and recurrent masks need a complete cross-product test matrix. QMIX's objective slicing is also transition-specific. The repaired transition-IQL path is not proof that all advertised multi-objective combinations work. |
| Medium | `algos/acer.py` | Its documented centralized extension mixes expected utilities. For nonlinear mixers this is not the expectation of joint Q. Marginal policy correction is not an exact joint-action correction. These are disclosed approximations, not bugs silently changed in this audit. |
| Medium | `algos/optimism/vbe.py` | Bonus computation compares predictor and target at the current observation, while training compares the predictor at current observations with target outputs at successor observations. The intended novelty definition needs to be made consistent. An empty bonus history also makes the update's `np.stack` fail. |
| Medium | `agents/mcts/mcts.py`, `node.py` | Adversarial search maximizes the same selection score at every player turn; cached subtree roots retain a parent. Reward/value perspective and root detachment need adversarial toy-game tests. Cooperative use does not validate the adversarial extension. |
| Medium | `runners/simple_runner.py` | Evaluation reseeds global Python/NumPy/Torch RNGs without restoring training RNG states, so evaluation scheduling affects subsequent training randomness. The recurrent-state correction does not isolate these RNG streams. |
| Medium | Replay wrappers/composition | Biased replay's inherited insertion hooks still bypass its intended behavior. Prioritized replay assumes an initially empty, one-insertion/one-item wrapped store; wrapping an accumulating n-step store requires explicit coordination. Scheduled alpha changes do not rebuild all existing priorities. |
| Medium | `intrinsic_reward/local_graph/local_graph.py` | Predicting an unseen edge divides by zero, the stated occurrence threshold is commented out, and undirected edge bookkeeping is orientation-sensitive. Tiny/empty graphs also need explicit clustering behavior. |
| Medium | Generic checkpoint/resume | DQN online/target collisions are repaired, but generic trainers can still save same-named networks to the same path. Network files do not constitute a full resume snapshot of optimizer, schedules, replay, and normalization state. |

These are not all covered by new failing tests: some fail at an earlier obsolete interface, and others require a defined algorithm contract before a numerical expected result is meaningful. They are listed to prevent the passing suite from being mistaken for certification of the whole repository.

## Optimization opportunities

No throughput benchmark or learning-curve experiment was run. The following rankings are based on the operations in the code; there are no claimed measured speedups.

| Priority | Opportunity | Expected benefit and how to validate |
|---|---|---|
| High | Replace random deque indexing in replay with an indexed ring store. | Deque access in the middle is linear in buffer length. Large DQN replay samples repeatedly traverse it. Preserve insertion/sample semantics and benchmark at realistic capacities; coordinate physical indexing with PER. |
| High | Cache episode tensors and slice them for PPO/PPOC minibatches. | `EpisodeBatch.get_minibatch()` reconstructs batches from episode objects, repeating NumPy conversion and host-to-device transfers. Transition batches already have a tensor-slicing path. Check that modified rewards, masks, and dynamic action metadata survive slicing. |
| High | Reduce CUDA-to-host synchronization in training logs. | PPO/ACER/DQN call `.item()` and convert arrays during updates. Accumulate detached statistics on device and transfer once per logging interval. Preserve KL early stopping, which needs a decision before the next update. |
| High | Avoid materializing all counterfactuals in social influence and LAIES. | Counterfactual tensors grow with actions, agents, time, and episode count. Chunking or batching alternatives can lower peak memory. LAIES's rejection loop also calls `.any()` repeatedly; direct sampling of nonfactual joint actions may avoid synchronization and repeated draws. Verify the sampling distribution first. |
| Medium | Vectorize ensemble/head computations. | VBE loops over full networks and QPLEX over attention kernels. Batched parameters or vectorized calls may improve GPU occupancy, but must retain independent noise and gradient semantics. |
| Medium | Avoid repeated state hypernetwork work for multiple QMIX objectives. | Weights depend on state, not the objective index. Compute them once and apply them across objective utilities. Validate objective-axis indexing and numerical equivalence first. |
| Medium | Batch environment inference across concurrent runs/environments. | The simple runner performs inference for one environment at a time; process-per-run execution duplicates model/runtime overhead. A vectorized collector is a larger change that must preserve episode boundaries and replay provenance. |
| Medium | Precompute shaping distances with a multi-source graph search. | `AgentDistanceShaping._precompute_distances()` runs a shortest-path query for each node/exit pair. One search from all exits can compute the same distances with much less repeated traversal. |
| Medium | Stabilize CSV metric schemas or write schema-versioned chunks. | A new metric column makes `CSVLogWriter._reformat()` read and rewrite the whole file. Avoid repeated whole-log rewrites while retaining compatibility with lazy result readers. |
| Medium | Consolidate result-query collection. | Run status and aggregate helpers execute separate lazy-frame collections. Collect independent queries together or cache results by file version, especially when the UI polls many runs. |
| Low | Revisit allocations only after profiling the full update. | The repository already contains fused target updates, transition packing, and pinned staging. Additional pinning, compilation, or vectorization can cost more than it saves for small CPU models. Measure representative workloads before adding more machinery. |

Two changes here also remove avoidable work: QATTEN uses one scalar per agent instead of an agent-by-agent diagonal matrix, and temporal return routines compute boundary masks once outside the reverse-time loop. Their motivation is supported by shape/algebra checks, not a timing result.

## Validation and interpretation

Commands used for the final checks:

```bash
timeout 60 .venv/bin/pytest -q --tb=short
git diff --name-only -- '*.py' | xargs .venv/bin/ruff check --isolated --select E9,F63,F7,F82
.venv/bin/ruff check tests/test_algorithm_audit.py --output-format concise
git diff --check
```

Results: **532 passed, 1 skipped in 7.51 seconds**. The focused static checks and whitespace check passed. The repository's broader lint rules still report pre-existing style issues in touched files; I did not claim a clean repository-wide lint run. Validation used the local Python/PyTorch environment and CPU regressions. CUDA transfer behavior, compiled execution, long-run convergence, and external environments such as SMAC were not independently certified.

The changes deliberately affect past numerical behavior. In particular, PPO/PPOC now do a full epoch of work, n-step replay exposes completed transitions sooner, target initialization is synchronized for directly owned trainers, and return/reward calculations differ at episode boundaries. Preserve the old code revision with existing logs and start fresh runs for scientific comparisons. A legacy DQN checkpoint whose online weights were already overwritten by its target cannot recover the lost online weights retroactively.

The highest-value next validation is a small seeded integration matrix over feed-forward/recurrent IQL, QMIX, QPLEX, IPPO, and ACER, including unequal episode lengths and time limits, followed by focused profiling of replay indexing, episode minibatch conversion, and logging synchronization. The open legacy implementations above should not be included as validated baselines until their contracts and tests are repaired.
