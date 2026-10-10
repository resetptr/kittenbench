"""Observer logging must preserve dynamics and record state at the mutation boundary."""
import json

from kittenbench.env import KittenEnv
from kittenbench.scoring import score


def test_logging_is_observer_only_and_does_not_change_dynamics():
    logged, plain = KittenEnv(seed=0, days=2), KittenEnv(seed=0, days=2, trace=False)
    assert logged.reset() == plain.reset()
    calls = [json.loads(line).get('call') for line in open('tests/fixtures/qwen__qwen3.5-9b_seed0_d2.agent.jsonl')]
    for call in calls[1:]:
        if call:
            assert logged.step(call) == plain.step(call)
        else:
            for env in (logged, plain):
                env.metrics.invalid_calls += 1
                env.steps += 1
    assert score(logged) == score(plain)
    assert logged.rng.getstate() == plain.rng.getstate()
    assert logged.cat_rng.getstate() == plain.cat_rng.getstate()
    assert plain.trace == []


def test_initial_state_is_deep_copied_and_object_mutations_precede_advance(tmp_path):
    env = KittenEnv(seed=0, days=2)
    env.reset()
    env.step({'name': 'go_to', 'arguments': {'room': 'kitchen'}})
    t = env.t
    env.step({'name': 'use', 'arguments': {'verb': 'fill', 'object': 'water_bowl'}})
    at_fill = next(r for r in env.trace if r['type'] == 'state' and r['t'] == t
                   and r['objects']['water_bowl']['props']['ml'] > 0)
    assert at_fill['reason'] == 'advance_start'
    assert env._initial_state['objects']['water_bowl']['props']['ml'] == 0
    env.world.add('mess', (1, 8), {'type': 'urine'})
    destination = tmp_path / 'trace.jsonl'
    env.save_trace(destination)
    meta = json.loads(destination.read_text().splitlines()[0])
    assert meta['trace_version'] == 4
    assert 'mess_1' not in meta['objects']
    assert meta['objects']['water_bowl']['props']['ml'] == 0


def test_forced_activity_records_end_of_interrupted_plan_and_new_event():
    env = KittenEnv(seed=0)
    previous = next(r for r in env.trace if r['type'] == 'cat')
    env.force_cat_step(env.cats[0], 'play_with_toy', 5, env.world.objects['toy_mouse'])
    stop = next(r for r in env.trace if r['type'] == 'cat_end')
    activity = [r for r in env.trace if r['type'] == 'cat'][-1]
    assert stop['event_id'] == previous['event_id']
    assert stop['t'] == env.t
    assert activity['t1'] == env.t + 5
    assert activity['t_arrive'] == env.t
    assert len(activity['path']) == 1
    assert activity['event_id'] != previous['event_id']


def test_owner_trace_records_target_and_actual_result_without_changing_observation():
    env = KittenEnv(seed=0)
    env.reset()
    env.step({'name': 'go_to', 'arguments': {'room': 'living_room'}})
    before = env.cats[0].tile
    obs = env.step({'name': 'inspect_cat', 'arguments': {'region': 'overall'}})
    event = [r for r in env.trace if r['type'] == 'owner'][-1]
    assert event['target'] == {'kind': 'cat', 'name': 'Miso', 'tile': before, 'room': 'living_room'}
    assert event['result'] in obs.text
    assert 'target' not in obs.data
    env.force_cat_step(env.cats[0], 'sleep_on_couch', 10, env.world.objects['couch'])
    obs = env.step({'name': 'play', 'arguments': {'minutes': 5}})
    event = [r for r in env.trace if r['type'] == 'owner'][-1]
    assert event['result'].startswith('You try to play, but ')
    assert event['t1'] - event['t0'] == 2
    assert event['result'] in obs.text
