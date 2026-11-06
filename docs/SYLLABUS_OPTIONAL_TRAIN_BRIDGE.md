# Optional Syllabus training-curriculum bridge (source compatibility, not deployment)

`labyrinth/syllabus_train_bridge.py` is a **TRAIN-only advisory adapter** to
[Syllabus](https://github.com/RyanNavillus/Syllabus) by Ryan Sullivan et al.
It does **not** replace Labyrinth's original regenerative model-running epic.
It does not load models, update weights, alter the canonical maze, order rooms,
change the Ariadne thread, authorize examination access, or qualify graduation.

## Audited upstream reference

- Source: `https://github.com/RyanNavillus/Syllabus`
- Inspected source commit: `2d3b9bdad95581ee2945499193f34dc2ef4c1507`
- Project packaging: `setup.py` reports version `0.7`; distribution
  [Syllabus-RL](https://pypi.org/project/Syllabus-RL/). Not installed by Labyrinth.
- License: MIT, `LICENSE` in that exact source checkout, copyright 2023 Ryan Sullivan.
  Preserve its notice if reusing upstream code. This adapter calls public APIs
  without copying the upstream implementation.
- Audited API files: `syllabus/core/curriculum_base.py` (`sample(k)`,
  `update_task_progress(task, progress)`,
  `update_on_episode(episode_return, length, task, progress)`),
  `syllabus/curricula/online_learning_progress.py`, and
  `syllabus/task_space/task_space.py` (`DiscreteTaskSpace`).

This is a **source-matched v0.7 protocol**, not proof that a particular PyPI
wheel, Python/Windows environment or upstream installation was run. Pin the
exact package/wheel and reverify upstream APIs before enabling it on hosts.

## Exact existing Labyrinth seam

```python
from labyrinth.syllabus_train_bridge import make_syllabus_train_bridge

# These are Labyrinth-owned, prevalidated source objects; in subsequent cycles
# use the current JourneyState's validated scope/maze, not fresh exam material.
bridge = make_syllabus_train_bridge(scope, maze_plan)  # optional package

# ONLY after the trusted development operator has finished a TRAIN chamber:
bridge.record_completed_train_room(maze_plan.rooms[0], correct=False, length=1)

# Non-authoritative next-practice suggestions. No mutation to MazePlan:
ideas = bridge.advisory_skill_sample(1)
```

`TrainingCorridorRunner.observations` can provide operator-side `room_id`,
`skill` and development correctness, while the canonical `MazeRoom` comes from
`JourneyState.maze.rooms`. The trusted operator must match exact room ID and
case admission and must never feed private examination data through this API.

The adapter maps `scope.skills` to numbered `DiscreteTaskSpace` entries. An
operator-completed TRAIN room emits **both** upstream progress and episode
updates; upstream `OnlineLearningProgress.update_on_episode` alone does not
accumulate per-task success. A sampled task ID is mapped back only to a declared
skill name. Duplicate room feedback, malformed IDs, nonboolean correctness,
bad samples, and replay after a partial upstream error are rejected.
The bridge consumes at most the requested sample count plus one sentinel;
even an accidental infinite upstream sample iterator cannot cause unbounded
materialization of task IDs.

**No protected exam influence:** `journey.py` still generates canonical future
mazes from a failed internal exam's `weak_skills` and the scope fingerprint.
Syllabus samples are *not* passed as those weak skills. Private Minotaur
qualification remains in the independent custody path; this helper has no
entry point to `blind_evaluation_custody.py` or `journey_qualification.py`.

## Evidence and limitations

Core test (does not require Syllabus):

```sh
python -m unittest tests.test_syllabus_train_bridge -v
```

The tests exercise the exact v0.7 method-call shapes through a synthetic
curriculum, optional import failure, replay protection, strict input checks,
and unchanged canonical maze identity. This does not prove real Syllabus
package interoperability or that adaptive suggestions improve reasoning.
Such claims require isolated pinned-package trials, learner parameter-delta
and optimizer receipts, closed-train feedback loops, contamination controls,
and withheld independent exams. No cloud spending is authorized or needed.

> **Hostile challenge:** An adaptive sampler can optimize a misleading local
> accuracy proxy and repeatedly expose near-identical tasks. Syllabus's
> sampling output is therefore never treated as independent evidence of model
> improvement. The real Labyrinth operator must measure novel, disjoint train
> transfers and actual parameter effects. A green bridge test only proves the
> protocol adapter, not learning or graduation.
