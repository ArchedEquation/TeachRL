"""tests/test_env.py — Full test suite for TeachRL v2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import pytest
import numpy as np
from env.archetypes import ArchetypeID, ALL_ARCHETYPES, make_archetype, CONCEPTS
from env.student import StudentSimulator
from env.environment import TeachRLEnv, TASK_REGISTRY, TutorAction
from env.gym_wrapper import TeachRLGymEnv
from graders.grader import EasyGrader, MediumGrader, HardGrader, ExpertGrader
from baseline.agents import RandomAgent, HeuristicAgent, GreedyArchetypeAgent, ArchetypeInferenceAgent
from self_play.escalator import SelfPlayEscalator, EpisodeResult

def simple_agent(obs): return {"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None}

class TestArchetypes:
    def test_all_8_load(self):
        rng = np.random.default_rng(0)
        for aid in ALL_ARCHETYPES:
            a = make_archetype(aid, rng)
            assert a.archetype_id == aid

    def test_bkt_params_valid(self):
        rng = np.random.default_rng(0)
        for aid in ALL_ARCHETYPES:
            a = make_archetype(aid, rng)
            for c in CONCEPTS:
                for d in ["easy","medium","hard"]:
                    p = a.get_params(c, d)
                    assert 0 < p.p_learn <= 1
                    assert 0 < p.p_forget <= 1

    def test_engagement_delta_bounded(self):
        rng = np.random.default_rng(0)
        for aid in ALL_ARCHETYPES:
            a = make_archetype(aid, rng)
            ed = a.engagement_delta("algebra_basics","medium",True,1,0.5)
            assert -0.5 <= ed <= 0.5

class TestStudentSimulator:
    def test_reset_all_archetypes(self):
        sim = StudentSimulator(seed=42)
        for aid in ALL_ARCHETYPES:
            sim.reset(seed=42, archetype_id=aid)
            assert sim.archetype_id == aid

    def test_answer_returns_bool_and_delta(self):
        sim = StudentSimulator(seed=42); sim.reset(seed=42)
        correct, delta = sim.answer_question("algebra_basics","easy")
        assert isinstance(correct, bool)
        assert isinstance(delta, float)

    def test_engagement_stays_bounded(self):
        sim = StudentSimulator(seed=0); sim.reset(seed=0)
        for _ in range(30):
            sim.answer_question("algebra_basics","hard")
        assert 0.05 <= sim.state.engagement <= 1.0

    def test_fatigue_increases(self):
        sim = StudentSimulator(seed=0); sim.reset(seed=0)
        for c in CONCEPTS: sim.answer_question(c,"easy")
        assert sim.state.fatigue > 0.0

class TestEnvironment:
    def test_all_tasks_reset(self):
        for task_id in TASK_REGISTRY:
            env = TeachRLEnv(task_id=task_id, seed=42)
            obs = env.reset()
            assert obs.engagement == 1.0
            assert obs.step_count == 0

    def test_step_returns_valid_reward(self):
        env = TeachRLEnv(task_id="blind_teaching", seed=42); env.reset()
        for _ in range(5):
            r = env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None})
            assert 0.0 <= r.reward <= 1.0
            if r.done: break

    def test_archetype_guess_reward_bonus(self):
        env = TeachRLEnv(task_id="blind_teaching", seed=42, eval_mode=True)
        env.reset()
        true_id = env._sim.archetype_id.value
        r_correct = env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":true_id})
        env.reset()
        r_wrong = env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":"impostor" if true_id!="impostor" else "adhd_sprinter"})
        assert r_correct.reward >= r_wrong.reward

    def test_task_scores_strictly_between_0_and_1(self):
        for task_id in TASK_REGISTRY:
            env = TeachRLEnv(task_id=task_id, seed=42)
            env.reset()
            for _ in range(env.task_cfg["max_steps"]):
                r = env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None})
                if r.done: break
            s = env._task_score()
            assert 0.0 < s < 1.0, f"{task_id}: score={s}"

    def test_done_after_max_steps(self):
        env = TeachRLEnv(task_id="archetype_identification", seed=42); env.reset()
        done = False; steps = 0
        while not done:
            r = env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None})
            done = r.done; steps += 1
        assert steps <= 21

    def test_reset_after_done(self):
        env = TeachRLEnv(task_id="archetype_identification", seed=42); env.reset()
        for _ in range(25):
            if env._done: break
            env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None})
        obs = env.reset()
        assert env._done is False and env._step_count == 0

    def test_state_api(self):
        env = TeachRLEnv(task_id="blind_teaching", seed=42, eval_mode=True); env.reset()
        s = env.state()
        assert s.task_id == "blind_teaching"
        assert s.true_archetype is not None

class TestSelfPlayEscalator:
    def test_escalation_triggers_after_good_scores(self):
        esc = SelfPlayEscalator()
        for _ in range(3):
            esc.record(EpisodeResult(ArchetypeID.OVERCONFIDENT,0.85,{},0.8,0.2,20,True))
        assert esc.generation(ArchetypeID.OVERCONFIDENT) >= 1

    def test_no_escalation_on_poor_scores(self):
        esc = SelfPlayEscalator()
        for _ in range(5):
            esc.record(EpisodeResult(ArchetypeID.ADHD,0.40,{},0.6,0.3,40,False))
        assert esc.generation(ArchetypeID.ADHD) == 0

    def test_weakest_archetype(self):
        esc = SelfPlayEscalator()
        # Record many episodes so ADHD and OVERCONFIDENT have history
        # All other archetypes remain at 0.0 — weakest is one of the unrecorded ones
        for _ in range(4):
            esc.record(EpisodeResult(ArchetypeID.OVERCONFIDENT,0.9,{},0.8,0.2,20,True))
            esc.record(EpisodeResult(ArchetypeID.ADHD,0.5,{},0.5,0.4,30,False))
        # Weakest should NOT be overconfident (score 0.9)
        weakest = esc.weakest_archetype()
        scores = esc.summary()["avg_scores"]
        assert scores[weakest.value] < scores[ArchetypeID.OVERCONFIDENT.value]

class TestGraders:
    def test_easy_grader(self):
        r = EasyGrader(n_episodes=3).grade(simple_agent)
        assert 0.0 < r.score < 1.0; assert r.episodes_run == 3

    def test_medium_grader(self):
        r = MediumGrader(n_episodes=3).grade(simple_agent)
        assert 0.0 < r.score < 1.0

    def test_hard_grader(self):
        r = HardGrader(n_episodes=3).grade(simple_agent)
        assert 0.0 < r.score < 1.0

    def test_expert_grader(self):
        r = ExpertGrader(n_episodes=2).grade(simple_agent)
        assert 0.0 < r.score < 1.0

class TestGymWrapper:
    def test_obs_shape(self):
        env = TeachRLGymEnv(task_id="blind_teaching",seed=42)
        obs,_ = env.reset()
        assert obs.shape == (45,) and obs.dtype == np.float32

    def test_step(self):
        env = TeachRLGymEnv(task_id="blind_teaching",seed=42); env.reset()
        obs,reward,done,trunc,info = env.step(env.action_space.sample())
        assert 0.0 <= reward <= 1.0

class TestBaselineAgents:
    def _run(self, agent, task_id="blind_teaching"):
        env = TeachRLEnv(task_id=task_id, seed=7); obs = env.reset(); done=False; steps=0
        while not done and steps<10:
            r=env.step(agent(obs.model_dump())); obs=r.observation; done=r.done; steps+=1
        return env._task_score()

    def test_random(self): assert 0<self._run(RandomAgent())<1
    def test_heuristic(self): assert 0<self._run(HeuristicAgent())<1
    def test_greedy(self): assert 0<self._run(GreedyArchetypeAgent())<1
    def test_inference(self): assert 0<self._run(ArchetypeInferenceAgent())<1

class TestReproducibility:
    def test_same_seed_same_result(self):
        def run(seed):
            env=TeachRLEnv(task_id="blind_teaching",seed=seed); env.reset(seed=seed); rewards=[]
            for _ in range(5):
                r=env.step({"concept":"algebra_basics","difficulty":"easy","hint_given":False,"archetype_guess":None})
                rewards.append(r.reward)
                if r.done: break
            return rewards
        assert run(42)==run(42)
